import json
import sqlite3
import logging
from typing import List, Dict
import os

class DataStorage:
    def __init__(self, db_path: str = "data/processed_data.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT,
                title TEXT,
                content TEXT,
                source TEXT,
                timestamp REAL,
                metadata TEXT
            )
        ''')
        
        # Create training_data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT,
                answer TEXT,
                context TEXT,
                source_doc_id INTEGER,
                FOREIGN KEY (source_doc_id) REFERENCES documents (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_documents(self, documents: List[Dict]) -> List[int]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        document_ids = []
        
        for doc in documents:
            cursor.execute('''
                INSERT INTO documents (url, title, content, source, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                doc.get('url', ''),
                doc.get('title', ''),
                doc.get('content', ''),
                doc.get('source', ''),
                doc.get('timestamp', 0),
                json.dumps(doc.get('metadata', {}))
            ))
            
            document_ids.append(cursor.lastrowid)
        
        conn.commit()
        conn.close()
        
        logging.info(f"Stored {len(documents)} documents in database")
        return document_ids
    
    def get_documents(self, limit: int = None) -> List[Dict]:
        """Retrieve documents from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM documents"
        if limit:
            query += f" LIMIT {limit}"
            
        cursor.execute(query)
        rows = cursor.fetchall()
        
        documents = []
        for row in rows:
            doc = {
                'id': row[0],
                'url': row[1],
                'title': row[2],
                'content': row[3],
                'source': row[4],
                'timestamp': row[5],
                'metadata': json.loads(row[6]) if row[6] else {}
            }
            documents.append(doc)
        
        conn.close()
        return documents
    
    def store_training_data(self, training_data: List[Dict]) -> List[int]:
        """Store training data (Q&A pairs) in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        training_ids = []
        
        for item in training_data:
            cursor.execute('''
                INSERT INTO training_data (question, answer, context, source_doc_id)
                VALUES (?, ?, ?, ?)
            ''', (
                item.get('question', ''),
                item.get('answer', ''),
                item.get('context', ''),
                item.get('source_doc_id', None)
            ))
            
            training_ids.append(cursor.lastrowid)
        
        conn.commit()
        conn.close()
        
        logging.info(f"Stored {len(training_data)} training examples in database")
        return training_ids
    
    def get_training_data(self) -> List[Dict]:
        """Retrieve training data from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM training_data")
        rows = cursor.fetchall()
        
        training_data = []
        for row in rows:
            item = {
                'id': row[0],
                'question': row[1],
                'answer': row[2],
                'context': row[3],
                'source_doc_id': row[4]
            }
            training_data.append(item)
        
        conn.close()
        return training_data

