# Docker 設置說明

本文檔說明如何使用 Docker Compose 啟動 Redis 和 PostgreSQL 服務，以及如何在應用程序中使用這些服務。

## 啟動服務

在項目根目錄中運行以下命令啟動 Redis 和 PostgreSQL 服務：

```bash
docker-compose up -d
```

這將在後台啟動服務。要查看服務日誌，可以運行：

```bash
docker-compose logs -f
```

## 服務配置

### Redis

- **容器名稱**：probalistic_trader_redis
- **端口**：6379（映射到主機的 6379）
- **數據卷**：redis_data（持久化數據）

### PostgreSQL

- **容器名稱**：probalistic_trader_postgres
- **端口**：5432（映射到主機的 5433）
- **數據卷**：postgres_data（持久化數據）
- **數據庫名稱**：nautilus_trader
- **用戶名**：nautilus
- **密碼**：password

## 在應用程序中使用

### 從主機訪問服務

如果您的應用程序在主機上運行（不在 Docker 容器中），請使用 DEFAULT_CONFIG：

```python
from database.config import DEFAULT_CONFIG

# 使用配置...
```

### 從 Docker 容器訪問服務

如果您的應用程序在 Docker 容器中運行，請使用 DOCKER_CONFIG：

```python
from database.config import DOCKER_CONFIG

# 使用配置...
```

## 管理命令

### 啟動服務

```bash
docker-compose up -d
```

### 停止服務

```bash
docker-compose down
```

### 查看服務狀態

```bash
docker-compose ps
```

### 查看服務日誌

```bash
docker-compose logs -f
```

### 連接到 PostgreSQL

```bash
docker exec -it probalistic_trader_postgres psql -U nautilus -d nautilus_trader
```

### 連接到 Redis

```bash
docker exec -it probalistic_trader_redis redis-cli
```
