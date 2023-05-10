from django.shortcuts import render
from django.http import HttpResponse
from Strength_Check.strong_password_generator import main
from Strength_Check.password_strength_analyser import main_function
# Create your views here.


"""
def index(request):
    return HttpResponse("Hello, world. You're at the index page.")
"""


def login(request):
    if request.method == "POST" and 'pswdgen' in request.POST:
        fullname = request.POST['full_name']
        emailId = request.POST['email']
        petName = request.POST['petName']
        DOB = request.POST['DOB']
        pswd_length = request.POST['length']
        generated_password = main(fullname, emailId, petName, DOB, pswd_length)

        return render(request, 'Generator_Classifier.html', {"generated_password": generated_password})
    elif request.method == "POST" and 'strencheck' in request.POST:
        password = request.POST['pswd']
        password_strength,color, message = main_function(password)
        return render(request, 'login.html', {"check_password": password_strength, "color":color,
                                                             "message": message})

    else:
        return render(request, 'login.html')


def generator_classifier(request):
    if request.method == 'POST' and 'checkstren' in request.POST:
        passwords = request.POST['pswds']
        password_strength, color, message = main_function(passwords)
        return render(request, 'Generator_Classifier.html', {"check_password": password_strength,
                                                             "color": color,
                                                             "message": message})

    return render(request, 'Generator_Classifier.html')