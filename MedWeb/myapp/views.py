from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .medbot.medbot import tree_to_code  # Import your bot functions
from django.http import HttpResponse
from rest_framework.views import APIView
from . models import *
from rest_framework.response import Response
from . serializer import *



def home(request):
    return HttpResponse("Hello, Django!")


class ReactView(APIView):
    def get(self, request):
        output = [{"input":output.input,
                   "output":output.output}
                  for output in React.objects.all()]
        return Response(output)
    
    def post(self, request):
        serializer = ReactSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data)




# def chat_view(request):
#     if request.method == 'POST':
#         user_input = request.POST.get('user_input')  # Assuming your user input is sent as a POST parameter
#         # Call the modified tree_to_code function with user input
#         bot_response = tree_to_code(your_decision_tree_model, your_feature_names, user_input)
#         return JsonResponse({'bot_response': bot_response})
#     else:
#         return render(request, 'chat.html')

