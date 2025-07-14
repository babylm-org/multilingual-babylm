"""
OpenSubtitles-specific functionality for downloading and processing subtitle data.
This module handles the OpenSubtitles dataset specifically.
"""

import os
import re
import shutil
import requests
import zipfile
import sqlite3
from pathlib import Path
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET
import pandas as pd
from typing import Dict, List, Tuple, Optional
from preprocessor import BasePreprocessor
import pandas as pd

BASE_URL = ("https://object.pouta.csc.fi/OPUS-OpenSubtitles/"
            "v2024/xml/{lang}.zip")


def filter_movies_by_id(folder_name: str, cursor, forbidden_genres: list = []) -> bool:
    """
    The IMDB ID of a subtitle file is the name of the folder in which it is contained. 
    (lang_id -> year_of_release -> imdb_id -> subtitle_file.xml)

    This function produces a boolean output by matching the imdb_id with certain conditions.
    1) True if the folder name has a valid number of digits is less than 10058830 (max movie ID contained in title.basics spreadsheet)
    2) True if the imdb_id corresponds to any genre not contained in the forbidden_genres list
    """
    MAX_DIGITS = 7
    num_digits = len(folder_name)

    if num_digits > MAX_DIGITS:
        return False

    imdb_id = "tt" + ("0"*(MAX_DIGITS - num_digits)) + folder_name

    genre_query = f"SELECT genres, isAdult FROM imdb WHERE tconst = '{imdb_id}';"
    
    result = cursor.execute(genre_query).fetchone()
    
    if result is None:
        # No movie found with this ID - you may want to include or exclude these
        return True  # or False, depending on your preference
    
    genres = result[0]  # Get the genres column
    is_adult = result[1]
    
    # Filter out adult content
    if bool(is_adult):
        return False
    
    if genres is None or genres == '\\N':  # Handle NULL or missing genres
        return True
    
    # Check if any genre is in forbidden list
    for g in genres.split(","):
        g = g.strip()  # Remove whitespace
        if g in forbidden_genres:
            return False
            
    return True


class OpenSubtitlesProcessor:
    """Handles downloading and processing OpenSubtitles datasets."""
    
    def __init__(self, lang_code: str, 
                 imdb_db_path: Path = Path("./imdb_mastersheet.db"), 
                 output_dir: Path = Path("./output"),
                 forbidden_genres: List[str] = None):
        self.lang_code = lang_code
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path("./temp_xml_batch")
        self.preprocessed_dir = output_dir / "preprocessed_texts"
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        # SQLite database setup
        self.imdb_db_path = imdb_db_path
        self.forbidden_genres = forbidden_genres or []
        
        # Verify database exists
        if not self.imdb_db_path.exists():
            raise FileNotFoundError(f"IMDB database not found at {self.imdb_db_path}")
            
    def _get_db_connection(self):
        """Get a database connection."""
        return sqlite3.connect(self.imdb_db_path)
        
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
    
    def filter_movies_by_id(self, folder_name: str, cursor) -> bool:
        """
        Check if a movie should be included based on IMDB data.
        Uses the standalone function with the instance's forbidden_genres.
        """
        return filter_movies_by_id(folder_name, cursor, self.forbidden_genres)

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
    
    def extract_sentences_from_xml(self, xml_path: Path) -> List[str]:
        """
        Return a list of plain-text sentences found in an OpenSubtitles TEI file.
        Works with/without namespaces and with/without <w> tokens.
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 1) normalise tag names
            for el in root.iter():
                if isinstance(el.tag, str) and el.tag.startswith('{'):
                    el.tag = el.tag.split('}', 1)[1]

            sentences: List[str] = []

            for s in root.iter('s'):
                # first preference: join <w> tokens (standard in OS 2024 dumps)
                tokens = [w.text for w in s.iter('w') if w.text]

                # fallback: take all text nodes inside <s>
                if not tokens:
                    tokens = [t.strip() for t in s.itertext() if t.strip()]

                if tokens:
                    sentences.append(' '.join(tokens))

            return sentences

        except Exception as e:
            print(f"Error extracting sentences from {xml_path}: {e}")
            return []
        

    def process_xml_to_text(self, xml_path: Path, output_path: Path, 
                           preprocessor: Optional['BasePreprocessor'] = None) -> bool:
        """Convert XML content to cleaned text format."""
        try:
            # Extract sentences
            sentences = self.extract_sentences_from_xml(xml_path)
            
            if not sentences:
                return False
            
            # Use preprocessor if provided, otherwise use default
            if preprocessor:
                processed_text = preprocessor.process_text('\n'.join(sentences).lower())
                # processed_sentences = preprocessor.preprocess_lines(sentences)
                # processed_text = '\n'.join(processed_sentences)
            else:
                # Fallback to simple cleaning - preserve structure
                processed_sentences = []
                for sentence in sentences:
                    if sentence == '':  # Preserve empty lines as paragraph breaks
                        processed_sentences.append('')
                    else:
                        cleaned = sentence.strip()
                        cleaned = re.sub(r'\s+', ' ', cleaned)  # Only normalize spaces, not newlines
                        if cleaned:
                            processed_sentences.append(cleaned)
                
                # Join with newlines, converting empty lines to paragraph breaks
                processed_text = []
                for i, sent in enumerate(processed_sentences):
                    if sent == '':
                        if processed_text and not processed_text[-1].endswith('\n\n'):
                            processed_text.append('\n')  # Add paragraph break
                    else:
                        processed_text.append(sent)
                        if i < len(processed_sentences) - 1 and processed_sentences[i + 1] != '':
                            processed_text.append('\n')  # Single newline between sentences
                
                processed_text = ''.join(processed_text)
            
            # Write output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(processed_text)
            
            return True
        except Exception as e:
            print(f"Error processing XML {xml_path}: {e}")
            return False
    
    def process_zip_in_batches(self, zip_path: Path, batch_size: int = 50,
                              preprocessor: Optional['BasePreprocessor'] = None,
                              enable_imdb_filtering: bool = True) -> pd.DataFrame:
        """Process XML files from zip in batches and return metadata DataFrame."""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_records = []
        
        # Get database connection if filtering is enabled
        conn = None
        cursor = None
        if enable_imdb_filtering:
            try:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                print(f"Connected to IMDB database. Filtering enabled with forbidden genres: {self.forbidden_genres}")
            except Exception as e:
                print(f"Warning: Could not connect to IMDB database: {e}")
                print("Proceeding without IMDB filtering...")
                enable_imdb_filtering = False
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                all_xml_members = [m for m in zip_file.namelist() if m.endswith('.xml')]
                
                # Tracking statistics
                total_files = len(all_xml_members)
                filtered_out_count = 0
                processed_count = 0
                
                for i in tqdm(range(0, len(all_xml_members), batch_size), 
                             desc="Processing XML Batches"):
                    batch_members = all_xml_members[i:i + batch_size]
                    current_batch_files = []
                    
                    for member_name in batch_members:
                        year, folder_id, file_id = self.extract_file_metadata(member_name)
                        
                        # Apply IMDB filtering if enabled
                        if enable_imdb_filtering and cursor:
                            if not self.filter_movies_by_id(folder_id, cursor):
                                filtered_out_count += 1
                                # Still record the filtered file in metadata
                                metadata_records.append({
                                    "file_id": file_id,
                                    "year": year,
                                    "folder_name": folder_id,
                                    "processing_status": "filtered_out",
                                    "filter_reason": "imdb_criteria"
                                })
                                continue
                        
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
                            success = self.process_xml_to_text(extracted_path, text_output_path, preprocessor)
                            
                            if success:
                                processed_count += 1
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
                
                # Print filtering statistics
                if enable_imdb_filtering:
                    print(f"\nFiltering Statistics:")
                    print(f"Total files: {total_files}")
                    print(f"Filtered out: {filtered_out_count}")
                    print(f"Successfully processed: {processed_count}")
                    print(f"Retention rate: {(processed_count/total_files)*100:.1f}%")
        
        finally:
            # Clean up database connection
            if conn:
                conn.close()
            
            # Final cleanup
            try:
                shutil.rmtree(self.temp_dir)
            except OSError:
                pass
        
        return pd.DataFrame(metadata_records)
    
    def process_language(self, batch_size: int = 50, keep_zip: bool = False,
                        preprocessor: Optional['BasePreprocessor'] = None,
                        enable_imdb_filtering: bool = True) -> Tuple[pd.DataFrame, Path]:
        """
        Complete pipeline to download and process a language's OpenSubtitles data.
        
        Args:
            batch_size: Number of files to process at once
            keep_zip: Whether to keep the downloaded zip file
            preprocessor: Optional preprocessor instance to use
            enable_imdb_filtering: Whether to apply IMDB-based filtering
        
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
        metadata_df = self.process_zip_in_batches(zip_dest, batch_size, preprocessor, enable_imdb_filtering)
        
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