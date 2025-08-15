from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
import torch
import json
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import time

# Инициализация FastAPI
app = FastAPI(title="GPT-OSS-20B API", description="OpenAI-compatible API for GPT-OSS-20B model")

# Загрузка модели и токенизатора
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model loaded on {device}")

# Хранилище API ключей (в реальном приложении лучше использовать безопасное хранилище)
API_KEYS = {
    "sk-example-key-1": {"user": "admin", "permissions": ["all"]},
    "sk-example-key-2": {"user": "user1", "permissions": ["chat", "completions"]},
}

# Настройка безопасности для API ключей
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Извлекаем ключ из заголовка "Bearer <key>"
    key = api_key.split("Bearer ")[-1] if "Bearer " in api_key else api_key
    
    if key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return key

# Модели Pydantic для валидации запросов и ответов
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "gpt-oss-20b"
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Dict[str, int]

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict]

# Функция для генерации ответа модели
async def generate_response(messages: List[Dict], max_tokens: int = 256, temperature: float = 1.0):
    try:
        # Применяем шаблон чата для токенизации
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        # Генерируем ответ
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Декодируем только сгенерированные токены
        response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        # Подсчитываем токены для usage
        input_tokens = inputs["input_ids"].shape[-1]
        output_tokens = outputs[0][inputs["input_ids"].shape[-1]:].shape[-1]
        
        return {
            "response_text": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# Эндпоинт для проверки работоспособности (без авторизации)
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Эндпоинт для чата (совместимый с OpenAI)
@app.post("/v1/chat/completions", response_model=Union[ChatCompletionResponse, ChatCompletionStreamResponse])
async def chat_completions(
    request: ChatCompletionRequest, 
    api_key: str = Depends(get_api_key)
):
    try:
        # Проверяем разрешения для ключа
        key_info = API_KEYS.get(api_key)
        if "chat" not in key_info["permissions"] and "all" not in key_info["permissions"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your API key doesn't have permission for this operation"
            )
        
        # Преобразуем сообщения в формат словаря
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Генерируем ответ
        result = await generate_response(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Формируем ответ в формате OpenAI
        response_id = f"cmpl-{torch.randint(0, 1000000, (1,)).item()}"
        created = int(time.time())
        
        if request.stream:
            # Возвращаем потоковый ответ
            async def generate_stream():
                # Отправляем первый чанк с контентом
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": result["response_text"]},
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                
                # Отправляем финальный чанк
                final_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            # Возвращаем обычный ответ
            response = ChatCompletionResponse(
                id=response_id,
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=result["response_text"])
                    )
                ],
                usage={
                    "prompt_tokens": result["input_tokens"],
                    "completion_tokens": result["output_tokens"],
                    "total_tokens": result["input_tokens"] + result["output_tokens"]
                }
            )
            return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Запуск приложения
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
