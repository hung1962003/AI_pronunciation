# Hướng dẫn chạy ứng dụng với FastAPI và Gunicorn

## Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## Chạy ứng dụng

### Cách 1: Chạy với Gunicorn + UvicornWorker (Production)

```bash
gunicorn webApp:app -c gunicorn_config.py
```

Hoặc chạy trực tiếp với các tham số:

```bash
gunicorn webApp:app --bind 0.0.0.0:8000 --workers 4 --worker-class uvicorn.workers.UvicornWorker
```

### Cách 2: Chạy với Uvicorn trực tiếp (Development)

```bash
uvicorn webApp:app --host 0.0.0.0 --port 8000
```

Hoặc chạy file trực tiếp:

```bash
python webApp.py
```

## Truy cập ứng dụng

Sau khi chạy, mở trình duyệt và truy cập:
- http://127.0.0.1:8000/

## Cấu hình Gunicorn

File `gunicorn_config.py` chứa các cấu hình:
- `bind`: Địa chỉ và port để bind (mặc định: 0.0.0.0:8000)
- `workers`: Số lượng worker processes (tự động tính dựa trên CPU cores)
- `worker_class`: Sử dụng UvicornWorker để hỗ trợ async
- `timeout`: Thời gian timeout cho mỗi request (30 giây)


