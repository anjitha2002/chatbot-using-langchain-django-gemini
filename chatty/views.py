import os
import json
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()


def chat_page(request):
    return render(request, 'chat.html')

@csrf_exempt
def chatbot_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST requests are allowed."}, status=405)

    if not os.getenv("GOOGLE_API_KEY"):
        return JsonResponse({"error": "GOOGLE_API_KEY not found in server environment."}, status=500)

    try:
        data = json.loads(request.body)
        user_message = data.get("message", "").strip()
        chat_history_data = data.get("history", [])
        
  
        allowed_models = [
            "gemini-3.8-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite"
        ]

        model_name = data.get("model", "allowed_models")

        if model_name not in allowed_models:
            return JsonResponse(
                {"error": "Invalid Gemini model selected."},
                status=400
            )
        temperature = float(data.get("temperature", 0.7))
        system_prompt = data.get("system_prompt", "You are a helpful assistant. Answer clearly and simply.")

        if not user_message:
            return JsonResponse({"error": "Message cannot be empty."}, status=400)

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        langchain_messages = [SystemMessage(content=system_prompt)]
        
        for msg in chat_history_data:
            if msg.get("role") == "user":
                langchain_messages.append(HumanMessage(content=msg.get("content")))
            elif msg.get("role") == "assistant":
                langchain_messages.append(AIMessage(content=msg.get("content")))

      
        langchain_messages.append(HumanMessage(content=user_message))

       
        response = llm.invoke(langchain_messages)
        reply = response.content

        if isinstance(reply,list):
                reply="\n".join(
                    block.get("text","")
                    if isinstance(block,dict)
                    else str(block)
                    for block in reply
                )
             
     
        return JsonResponse({"reply": reply})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


