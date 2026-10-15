# run.py
import sys
import os

# Thêm thư mục src vào sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )