"""
OpenSubtitles-specific functionality for downloading and processing subtitle data.
This module handles the OpenSubtitles dataset specifically.
"""

import os
import re
import shutil
import requests
import zipfile
from pathlib import Path
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET
import pandas as pd
from typing import Dict, List, Tuple, Optional

BASE_URL = ("https://object.pouta.csc.fi/OPUS-OpenSubtitles/"
            "v2024/xml/{lang}.zip")


class OpenSubtitlesProcessor:
    """Handles downloading and processing OpenSubtitles datasets."""
    
    def __init__(self, lang_code: str, output_dir: Path = Path("./output")):
        self.lang_code = lang_code
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path("./temp_xml_batch")
        self.preprocessed_dir = output_dir / "preprocessed_texts"
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
    def download_zip(self, dest: Path) -> Path:
        """Stream-download the language zip to dest and return the Path."""
        url = BASE_URL.format(lang=self.lang_code)
        dest.parent.mkdir(parents=True, exist_ok=True)

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))

            bar = tqdm(total=total, unit="B", unit_scale=True,
                      desc=f"⬇ {self.lang_code}.zip")

            with dest.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MiB
                    f.write(chunk)
                    bar.update(len(chunk))
            bar.close()

        return dest
    
    def extract_file_metadata(self, member_name: str) -> Tuple[str, str, str]:
        """Extract year, folder_id, and file_id from the zip member path."""
        parts = Path(member_name).parts
        
        try:
            # Find the index of 'xml' followed by language code
            xml_lang_index = -1
            for idx, part in enumerate(parts):
                if part == 'xml' and idx + 1 < len(parts) and parts[idx + 1] == self.lang_code:
                    xml_lang_index = idx + 1
                    break

            if xml_lang_index != -1 and len(parts) > xml_lang_index + 2:
                year = parts[xml_lang_index + 1]
                folder_id = parts[xml_lang_index + 2]
                file_id = Path(member_name).stem
            else:
                year = "Unknown"
                folder_id = "Unknown"
                file_id = Path(member_name).stem
        except IndexError:
            year = "Unknown"
            folder_id = "Unknown"
            file_id = Path(member_name).stem
            
        return year, folder_id, file_id
    
    def extract_xml_metadata(self, xml_path: Path) -> Dict:
        """Extract metadata from the <meta> tag in the XML file."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            meta = root.find("meta")
            if meta is not None:
                meta_data = {}
                for section in meta:
                    section_data = {}
                    for child in section:
                        section_data[child.tag] = child.text
                    meta_data[section.tag] = section_data
                return meta_data
            else:
                return {}
        except Exception as e:
            print(f"Error extracting metadata from {xml_path}: {e}")
            return {}
    
    def process_xml_to_text(self, xml_path: Path, output_path: Path) -> bool:
        """Convert XML content to cleaned text format."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            with open(output_path, 'w', encoding='utf-8') as txt_file:
                def process_element(element):
                    if element.tag == "meta":
                        return
                    
                    if element.tag == "w" and element.text:
                        # Clean and write word
                        cleaned_text = self.clean_text(element.text)
                        if cleaned_text:
                            txt_file.write(cleaned_text + " ")
                    
                    for child in element:
                        process_element(child)
                    
                    if element.tag == "s":
                        txt_file.write("\n")
                
                process_element(root)
            
            return True
        except Exception as e:
            print(f"Error processing XML {xml_path}: {e}")
            return False
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text
    
    def process_zip_in_batches(self, zip_path: Path, batch_size: int = 50) -> pd.DataFrame:
        """Process XML files from zip in batches and return metadata DataFrame."""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_records = []
        
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            all_xml_members = [m for m in zip_file.namelist() if m.endswith('.xml')]
            
            for i in tqdm(range(0, len(all_xml_members), batch_size), 
                         desc="Processing XML Batches"):
                batch_members = all_xml_members[i:i + batch_size]
                current_batch_files = []
                
                for member_name in batch_members:
                    year, folder_id, file_id = self.extract_file_metadata(member_name)
                    
                    # Extract file
                    extracted_path = self.temp_dir / member_name
                    extracted_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        with zip_file.open(member_name) as source, \
                             extracted_path.open("wb") as target:
                            shutil.copyfileobj(source, target)
                        current_batch_files.append(extracted_path)
                        
                        # Extract XML metadata
                        xml_metadata = self.extract_xml_metadata(extracted_path)
                        
                        # Process to text
                        text_output_path = self.preprocessed_dir / f"{file_id}.txt"
                        success = self.process_xml_to_text(extracted_path, text_output_path)
                        
                        if success:
                            # Create metadata record
                            record = {
                                "file_id": file_id,
                                "year": year,
                                "folder_name": folder_id,
                                "text_file_path": str(text_output_path),
                                "processing_status": "success"
                            }
                            
                            # Add XML metadata fields
                            for section, data in xml_metadata.items():
                                for key, value in data.items():
                                    record[f"meta_{section}_{key}"] = value
                            
                            metadata_records.append(record)
                        else:
                            metadata_records.append({
                                "file_id": file_id,
                                "year": year,
                                "folder_name": folder_id,
                                "processing_status": "failed"
                            })
                            
                    except Exception as e:
                        print(f"Error processing {member_name}: {e}")
                        continue
                
                # Clean up batch files
                for fpath in current_batch_files:
                    try:
                        os.remove(fpath)
                    except OSError as e:
                        print(f"Error deleting file {fpath}: {e}")
                
                # Clean up empty directories
                for p in sorted(self.temp_dir.glob('**/*'), 
                               key=lambda p: len(p.parts), reverse=True):
                    if p.is_dir() and not os.listdir(p):
                        try:
                            os.rmdir(p)
                        except OSError:
                            pass
        
        # Final cleanup
        try:
            shutil.rmtree(self.temp_dir)
        except OSError:
            pass
        
        return pd.DataFrame(metadata_records)
    
    def process_language(self, batch_size: int = 50, keep_zip: bool = False) -> Tuple[pd.DataFrame, Path]:
        """
        Complete pipeline to download and process a language's OpenSubtitles data.
        
        Returns:
            Tuple of (metadata_df, preprocessed_texts_dir)
        """
        zip_dest = Path(f"./datasets/{self.lang_code}.zip")
        
        # Download if not exists
        if not zip_dest.exists():
            print(f"Downloading {self.lang_code} dataset...")
            self.download_zip(zip_dest)
        
        # Process
        print(f"Processing {self.lang_code} dataset...")
        metadata_df = self.process_zip_in_batches(zip_dest, batch_size)
        
        # Save metadata
        metadata_path = self.output_dir / f"{self.lang_code}_file_metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        print(f"Metadata saved to {metadata_path}")
        
        # Clean up zip if requested
        if not keep_zip:
            try:
                os.remove(zip_dest)
                print(f"Deleted zip file: {zip_dest}")
            except OSError as e:
                print(f"Error deleting zip file: {e}")
        
        return metadata_df, self.preprocessed_dir