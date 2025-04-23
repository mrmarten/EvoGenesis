# EvoGenesis Docker Deployment Guide

This guide explains how to run EvoGenesis using Docker for both development and production environments.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed
- Git (for cloning the repository)

## Quick Start

1. Clone the repository (if not already done):
   ```
   git clone <your-repository-url>
   cd evoorg
   ```

2. Start EvoGenesis in production mode:
   ```
   docker-compose up -d
   ```

3. Access the web interface at: http://localhost:5000

## Deployment Options

### Production Deployment

For a production deployment with proper data persistence:

```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

The production deployment:
- Maps port 5000 for the web UI
- Persists data in the ./data directory
- Persists outputs in the ./output directory
- Uses configuration from ./config/default.json

### Development Environment

For active development with live code changes:

```bash
# Build and start the development container
docker-compose --profile dev up -d evogenesis-dev

# View logs
docker-compose logs -f evogenesis-dev

# Stop the container
docker-compose --profile dev down
```

The development environment:
- Maps port 5001 for the web UI (access at http://localhost:5001)
- Mounts the entire project directory for live code editing
- Runs in development mode with enhanced debugging

## Configuration

The default configuration is loaded from `config/default.json`. You can modify this file to change:

- LLM providers and models
- Web UI settings (port, host, etc.)
- Agent configuration
- Memory and vector store settings
- Security settings

## Data Management

EvoGenesis stores persistent data in these directories:

- `./data`: Vector store embeddings, agent memory, and other persistent data
- `./output`: Generated files and outputs from agent activities

These directories are persisted even when containers are removed.

## Troubleshooting

### Container fails to start

1. Check logs for errors:
   ```
   docker-compose logs -f
   ```

2. Verify port availability:
   ```
   netstat -tuln | grep 5000
   ```

3. Check file permissions:
   ```
   chmod -R 755 ./data ./output ./config
   ```

### Web UI is not accessible

1. Confirm container is running:
   ```
   docker-compose ps
   ```

2. Check container logs:
   ```
   docker-compose logs -f
   ```

3. Verify network settings:
   ```
   docker network inspect evoorg_default
   ```

## Advanced Usage

### Scaling with Docker Swarm

For distributed deployments:

```bash
# Initialize swarm
docker swarm init

# Deploy as a stack
docker stack deploy -c docker-compose.yml evogenesis
```

### Custom Configuration

Create a custom configuration file:

1. Copy the default configuration:
   ```
   cp config/default.json config/custom.json
   ```

2. Edit `config/custom.json` with your settings

3. Start with custom config:
   ```
   docker-compose run -e CONFIG_PATH=/app/config/custom.json evogenesis
   ```

## Updating EvoGenesis

To update to the latest version:

```bash
# Pull the latest code
git pull

# Rebuild and restart containers
docker-compose down
docker-compose build
docker-compose up -d
```
