# GPT-OSS-20B API Server

OpenAI-compatible API server for GPT-OSS-20B model built with FastAPI. This project allows you to deploy your own GPT-OSS-20B model and access it through an API that's compatible with OpenAI's API format.

## Features

- OpenAI-compatible API endpoints
- API key authentication for security
- Support for both streaming and non-streaming responses
- Easy integration with ProTalk bot and other OpenAI-compatible clients
- Configurable model parameters (temperature, max tokens, etc.)

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Server](#running-the-server)
  - [API Documentation](#api-documentation)
  - [Integration with ProTalk Bot](#integration-with-protalk-bot)
- [API Reference](#api-reference)
  - [Authentication](#authentication)
  - [Endpoints](#endpoints)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- Python 3.8 or higher
- PyTorch (compatible with your system)
- A VDS server with at least 2 CPU cores, 4GB RAM, and 40GB disk space
- Ubuntu 22.04 (recommended) or another Linux distribution

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/gpt-oss-20b-api.git
   cd gpt-oss-20b-api
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` doesn't exist, install the dependencies manually:
   ```bash
   pip install fastapi uvicorn torch transformers pydantic
   ```

## Usage

### Running the Server

To run the server in development mode:

```bash
python app.py
```

For production deployment, it's recommended to use a process manager like systemd:

1. Create a service file:
   ```bash
   sudo nano /etc/systemd/system/gpt-api.service
   ```

2. Add the following content (adjust paths as needed):
   ```
   [Unit]
   Description=GPT API Server
   After=network.target

   [Service]
   User=your_username
   WorkingDirectory=/path/to/gpt-oss-20b-api
   ExecStart=/path/to/gpt-oss-20b-api/venv/bin/python app.py
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

3. Enable and start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable gpt-api
   sudo systemctl start gpt-api
   ```

### API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Integration with ProTalk Bot

To integrate this API with the ProTalk bot:

1. Open the ProTalk bot
2. Go to API settings
3. Fill in the following fields:
   - **API Key**: `sk-example-key-1` (or any other key defined in the code)
   - **Model Name**: `gpt-oss-20b`
   - **Base URL**: `http://your-server-ip:8000/v1`

4. Save the settings and test the bot

## API Reference

### Authentication

All API requests must include an API key in the `Authorization` header:

```
Authorization: Bearer sk-example-key-1
```

### Endpoints

#### Health Check

Check if the server is running.

**Request:**
```http
GET /health
```

**Response:**
```json
{
  "status": "ok"
}
```

#### Chat Completions

Create a chat completion.

**Request:**
```http
POST /v1/chat/completions
Content-Type: application/json
Authorization: Bearer sk-example-key-1

{
  "model": "gpt-oss-20b",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "temperature": 1.0,
  "max_tokens": 256,
  "stream": false
}
```

**Response (non-streaming):**
```json
{
  "id": "cmpl-123456",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-oss-20b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I'm doing well, thank you for asking! How can I assist you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 20,
    "total_tokens": 32
  }
}
```

**Response (streaming):**
```
data: {"id": "cmpl-123456", "object": "chat.completion.chunk", "created": 1677652288, "model": "gpt-oss-20b", "choices": [{"index": 0, "delta": {"content": "I'm"}, "finish_reason": null}]}

data: {"id": "cmpl-123456", "object": "chat.completion.chunk", "created": 1677652288, "model": "gpt-oss-20b", "choices": [{"index": 0, "delta": {"content": " doing"}, "finish_reason": null}]}

...

data: {"id": "cmpl-123456", "object": "chat.completion.chunk", "created": 1677652288, "model": "gpt-oss-20b", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]
```

## Configuration

### API Keys

API keys are defined in the `API_KEYS` dictionary in `app.py`. You can add or modify keys as needed:

```python
API_KEYS = {
    "sk-example-key-1": {"user": "admin", "permissions": ["all"]},
    "sk-example-key-2": {"user": "user1", "permissions": ["chat", "completions"]},
}
```

### Model Parameters

You can adjust the model parameters in the `generate_response` function:

- `max_new_tokens`: Maximum number of tokens to generate
- `temperature`: Controls randomness (1.0 is default, higher values mean more random)
- `do_sample`: Whether to use sampling (set to True when temperature > 0)

## Deployment

### VDS Deployment

For deployment on a VDS server, follow these steps:

1. Connect to your VDS using SSH
2. Update the system: `sudo apt update && sudo apt upgrade -y`
3. Install Python and other prerequisites
4. Clone this repository
5. Create a virtual environment and install dependencies
6. Configure the firewall to allow port 8000: `sudo ufw allow 8000`
7. Set up the systemd service as described in the [Running the Server](#running-the-server) section

### Docker Deployment

You can also deploy this server using Docker:

1. Create a Dockerfile:
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   EXPOSE 8000

   CMD ["python", "app.py"]
   ```

2. Build the Docker image:
   ```bash
   docker build -t gpt-oss-20b-api .
   ```

3. Run the container:
   ```bash
   docker run -d -p 8000:8000 gpt-oss-20b-api
   ```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**: If you encounter memory errors, try reducing the `max_tokens` parameter or consider using a server with more RAM.

2. **Slow Response Times**: For better performance, ensure you're using a GPU-enabled server and that PyTorch is configured to use CUDA.

3. **Connection Refused**: Make sure the firewall is properly configured to allow connections on port 8000.

4. **Permission Denied**: Ensure the user running the application has the necessary permissions to access the model files and bind to the port.

### Logs

For systemd deployment, you can view logs with:
```bash
sudo journalctl -u gpt-api -f
```

For Docker deployment:
```bash
docker logs <container_id>
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
