from django import forms
from django.contrib.auth.models import User

class UserSelection(forms.Form):
    selected_user = forms.ModelChoiceField(label='Selecionar Usuario', queryset=User.objects.all(), required=True)