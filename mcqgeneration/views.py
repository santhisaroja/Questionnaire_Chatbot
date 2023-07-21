
from django.shortcuts import render,redirect
from django.views import generic
from django.views.generic import View
from django.views.generic.edit import CreateView,UpdateView,DeleteView
from django.http import HttpResponse 
from django.contrib.auth import authenticate,login
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.urls import reverse
from django.http import HttpResponseRedirect
from django.contrib import auth
from django.contrib.auth.hashers import make_password
import itertools
from django.http import HttpResponse
import PyPDF2
from PyPDF2 import PdfReader
import spacy
import string
import nltk
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from textwrap3 import wrap
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import string
import pke
import traceback
import random
from flashtext import KeywordProcessor
import numpy as np
from sense2vec import Sense2Vec
from sentence_transformers import SentenceTransformer
from collections import OrderedDict
from collections import OrderedDict
import requests
from bs4 import BeautifulSoup
# Importing neccessary models.
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user
from mcqgeneration.models import Chat
from .forms import SignUpForm
from mcqgeneration.models import *
import openai
import pandas as pd
import os
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .forms import DocumentForm


def base(request):
	return render(request,'base.html')




def upload_document(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            # Delete all previous documents' files
            for document in Document.objects.all():
                if os.path.exists(document.pdf_file.path):
                    os.remove(document.pdf_file.path)
                document.delete()

            # Save the new document
            document = form.save(commit=False)
            document.save()

            return render(request, 'doc_upload.html')
    else:
        form = DocumentForm()

    return render(request, 'pdf_upload.html', {'form': form})






def read_pdf(request):
	# Read the PDF file and extract text
	pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pdfs')
	files = os.listdir(pdf_path)
	last_file = os.path.join(pdf_path, files[0])
	with open(last_file, 'rb') as pdf_file:
		pdf_reader = PyPDF2.PdfReader(pdf_file)
		text = ""
		for page in range(len(pdf_reader.pages)):
			page_obj = pdf_reader.pages[page]
			text += page_obj.extract_text()

	
	# # summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
	# # summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')
	# summary_model = T5ForConditionalGeneration.from_pretrained('../MCQ_gen/mcqgen/mcqgeneration/models')
	# summary_tokenizer = T5Tokenizer.from_pretrained('../MCQ_gen/mcqgen/mcqgeneration/models')
	summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
	summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	summary_model = summary_model.to(device)

	# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# summary_model = summary_model.to(device)
	def postprocesstext (content):
		final=""
		for sent in sent_tokenize(content):
			sent = sent.capitalize()
			final = final +" "+sent
		return final 


	def summarizer(text,model,tokenizer):
		text = text.strip().replace("\n"," ")
		text = "summarize: "+text
		max_length = 512
		encoding = tokenizer.encode_plus(text,max_length=max_length, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)
		input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
		outs = model.generate(input_ids=input_ids,
	                                  attention_mask=attention_mask,
	                                  early_stopping=True,
	                                  num_beams=5,
	                                  num_return_sequences=1,
	                                  no_repeat_ngram_size=2,
	                                  min_length = 75,
	                                  max_length=500)
		dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
		summary = dec[0]
		summary = postprocesstext(summary)
		summary= summary.strip()
		return summary
	summarized_text = summarizer(text,summary_model,summary_tokenizer)


	def get_nouns_multipartite(content):
	    out=[]
	    try:
	        extractor = pke.unsupervised.MultipartiteRank()
	        extractor.load_document(input=content,language='en')
	        #    not contain punctuation marks or stopwords as candidates.
	        pos = {'PROPN','NOUN'}
	        #pos = {'PROPN','NOUN'}
	        stoplist = list(string.punctuation)
	        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
	        stoplist += stopwords.words('english')
	        # extractor.candidate_selection(pos=pos, stoplist=stoplist)
	        extractor.candidate_selection(pos=pos)
	        # 4. build the Multipartite graph and rank candidates using random walk,
	        #    alpha controls the weight adjustment mechanism, see TopicRank for
	        #    threshold/method parameters.
	        extractor.candidate_weighting(alpha=1.1,
	                                      threshold=0.75,
	                                      method='average')
	        keyphrases = extractor.get_n_best(n=15)
	        

	        for val in keyphrases:
	            out.append(val[0])
	    except:
	    	out = []
	    	traceback.print_exc()
	    return out
	def get_keywords(originaltext,summarytext):
		keywords = get_nouns_multipartite(originaltext)
		keyword_processor = KeywordProcessor()
		for keyword in keywords:
			keyword_processor.add_keyword(keyword)
		keywords_found = keyword_processor.extract_keywords(summarytext)
		keywords_found = list(set(keywords_found))
		important_keywords =[]
		for keyword in keywords:
			if keyword in keywords_found:
				important_keywords.append(keyword)
		return important_keywords[:20]

	imp_keywords = get_keywords(text,summarized_text)
	a = get_nouns_multipartite(text)
	question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
	question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
	# question_model.save_pretrained('../MCQ_gen/mcqgen/mcqgeneration/models')
	# question_tokenizer.save_pretrained('../MCQ_gen/mcqgen/mcqgeneration/models')
	# question_model = T5ForConditionalGeneration.from_pretrained('../MCQ_gen/mcqgen/mcqgeneration/models')
	# question_tokenizer = T5Tokenizer.from_pretrained('../MCQ_gen/mcqgen/mcqgeneration/models')
	question_model = question_model.to(device)
	def get_question(context,answer,model,tokenizer):
		text = "context: {} answer: {}".format(context,answer)
		encoding = tokenizer.encode_plus(text,max_length=384, pad_to_max_length=False,truncation=True, return_tensors="pt").to(device)
		input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
		outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=72)
		dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
		Question = dec[0].replace("question:","")
		Question= Question.strip()
		return Question
	for answer in imp_keywords:
		ques = get_question(summarized_text,answer,question_model,question_tokenizer)

	s2v = Sense2Vec().from_disk('s2v_old')
	# paraphrase-distilroberta-base-v1
	sentence_transformer_model = SentenceTransformer('msmarco-distilbert-base-v3')
	# sentence_transformer_model.save('../MCQ_gen/mcqgen/mcqgeneration/models')
	# sentence_transformer_model = SentenceTransformer('../MCQ_gen/mcqgen/mcqgeneration/models')
	def sense2vec_get_word(word,s2v):
		output = []
		word = word.lower()
		word = word.replace(" ", "_")
		sense = s2v.get_best_sense(word)
		most_similar = s2v.most_similar(sense, n=20)
		for each_word in most_similar:
			append_word = each_word[0].split("|")[0].replace("_", " ").lower()
			if append_word.lower() != word:
				output.append(append_word.title())
		out = list(OrderedDict.fromkeys(output))
		return out

		
	word = list(imp_keywords)
	
	
	np =  get_keywords(text,summarized_text)

	questions_list=[]
	answers_list=[]
	distractors_list=[]
	for answer in np:
		ques = get_question(summarized_text,answer,question_model,question_tokenizer)
		# distractors = sense2vec_get_word(answer,s2v)
		questions_list.append(ques)
		answers_list.append(answer)
	# 	dist=random.sample(distractors, 3)
	# 	distractors_list.append(dist)
	# # csv generation
	s_list =[]
	for i in answers_list:
		s_list.append(i.capitalize())
	df = pd.DataFrame()
	df['Question'] = questions_list
	df['Correct Answer'] = s_list	
	df.to_csv("question_ans.csv",index=False)
	#code gen 
	questions = questions_list
	# distractors = distractors_list
	answers = answers_list

	mcqs = []
	for i in range(len(questions)):
		q = {}
		q['text'] = questions[i]
		# q['options'] = [answers[i].capitalize()] + distractors[i]
		# random.shuffle(q['options'])
		q['correct_answer'] = answers[i].capitalize()
		mcqs.append(q)
	# random.shuffle(mcqs)
	return render(request, 'mcqs_gen.html', {'mcqs': mcqs})


	

def mcq_result(request):
	# mcqs = mcqs_renders()
	# ans_lst = []
	# for i in mcqs:
	# 	ans_lst.append(i['correct_answer'])
	if request.method == 'POST':
		selected_options = []
		df1 = pd.read_csv('question_ans.csv')
		correct_list = df1['Correct Answer'].to_list()
		for i in range(len(correct_list)):
			selected_option = request.POST.get('mcq' + str(i) + '[]')
			selected_options.append(selected_option)

		score = 0
		for i in selected_options:
			for j in correct_list:
				if i == j:
					score += 1
		

		df1['Your Answers'] = selected_options
		df1.index = range(1, len(df1)+1)
		html_table = df1.to_html()
		return render(request, 'mcq_result.html', {'score':score,'html_table':html_table})
# Login View
def signin(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('base')
        return render(request, 'signin.html', {'error': 'Invalid login credentials'})
    else:
        return render(request, 'signin.html')

# Signup view
def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            user.save()
            login(request, user)
            return redirect('signin')
    else:
        form = SignUpForm()
    return render(request, 'signup.html', {'form': form})
# Logout View
def logout_view(request):
    logout(request)
    return redirect('signin')
    
openai.api_key = "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"
# generating response from openai
def generate_response(user_input):
    prompt = (f"User: {user_input}")
    completions = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        #stop=None,
        temperature=0.5,
    )
    message = completions.choices[0].text
    return message

@login_required
def AiChat(request):
    # Form verification
    user = get_user(request)
    if request.method == 'POST':
        if request.user.is_authenticated:
            user_input = request.POST.get('user_input')
            ai_response = generate_response(user_input)
            # Saving both user_input and ai_response in the database            
            Chat.objects.create(user=user, user_input=user_input, ai_response=ai_response)
            chat_history = Chat.objects.filter(user=user)
            context = {
                'user_input':user_input,
                'chatbot_response':ai_response,
                'chat_history':chat_history
            }
            return render(request,'index.html', context)
    else:
        user_input = ""
        ai_response = ""
        chat_history = Chat.objects.filter(user=user)
        context = {
                'user_input':user_input,
                'chatbot_response':ai_response,
                'chat_history':chat_history
            }
        return render(request, 'index.html',context)
# Clear chat
def clear(request):
    # clear the 'name' field for all records in the 'MyModel' model
    Chat.objects.all().delete()
#    cache.clear()
    return redirect('index')
