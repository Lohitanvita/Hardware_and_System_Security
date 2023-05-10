from django.urls import path
from Strength_Check import views


urlpatterns = [
    path("", views.login, name="login"),
    path("gen_clf/", views.generator_classifier, name="Generator_Classifier")

]