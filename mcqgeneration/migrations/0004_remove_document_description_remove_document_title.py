# Generated by Django 4.1.7 on 2023-03-29 09:20

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('mcqgeneration', '0003_document_delete_pdfdocument'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='document',
            name='description',
        ),
        migrations.RemoveField(
            model_name='document',
            name='title',
        ),
    ]
