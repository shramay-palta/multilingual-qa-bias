from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AdamW
from transformers import get_linear_schedule_with_warmup, get_constant_schedule
import torch
import torch.nn as nn
import torch.nn.functional as F 
from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import math
from tqdm import tqdm
import copy
import pickle
import os
import random
import time

def find_answer(question_text, answer, ref_text, tokenizer) -> bool:
    model_input = tokenizer(question_text, ref_text, truncation=True, padding=True, return_token_type_ids=True, add_special_tokens=True)['input_ids'] #model input has both question text and reference text
    answer = tokenizer(answer, truncation=True, padding=True, return_token_type_ids=True, add_special_tokens=False)['input_ids']
    if len(model_input) > 512:
        return -1, -1
    for i in range(0, len(model_input) - len(answer) + 1):
        if answer == model_input[i: i + len(answer)]:
            return (i, i + len(answer))
    return -1, -1
# prepair input
def prepare_inputs(indexes, data, tokenizer):
    contexts = []
    questions = []
    answer_starts = []
    answer_ends = []
    for i in indexes:
        t = data[i]
        question = t['question']
        answer = t['answers']['text'][0]
        context = t['context']
        s, e = find_answer(question, answer, context, tokenizer)
        if s == -1:
            continue
        contexts.append(context)
        questions.append(question)
        answer_starts.append(s)
        answer_ends.append(e)
    return contexts, questions, answer_starts, answer_ends

def get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
    """
    Returns the correct learning rate scheduler
    """
    scheduler = scheduler.lower()
    if scheduler=='constantlr':
        return get_constant_schedule(optimizer)
    elif scheduler=='warmupconstant':
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    elif scheduler=='warmuplinear':
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    elif scheduler=='warmupcosine':
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    elif scheduler=='warmupcosinewithhardrestarts':
        return get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    else:
        raise ValueError("Unknown scheduler {}".format(scheduler))
        
def train(model, optimizer, scheduler, train_data, dev_data, batch_size, device, max_grad_norm, tokenizer, best_acc = -1):
    loss_fn = nn.CrossEntropyLoss()

    step_cnt = 0
    #best_model_weights = None
    
    contexts, questions, answer_starts, answer_ends = train_data

    for pointer in tqdm(range(0, len(contexts), batch_size), desc='training',ascii = True,leave = True):
        model.train() # model was in eval mode in evaluate(); re-activate the train mode
        optimizer.zero_grad() # clear gradients first
        torch.cuda.empty_cache() # releases all unoccupied cached memory 
        # t = torch.cuda.get_device_properties(0).total_memory
        # r = torch.cuda.memory_reserved(0) 
        # a = torch.cuda.memory_allocated(0)
        # print(t,r,a)

        step_cnt += 1
        
        input = tokenizer(contexts[pointer:min(pointer + batch_size, len(contexts))], questions[pointer:min(pointer + batch_size, len(contexts))], return_tensors="pt",truncation=True, padding=True, return_token_type_ids=True, add_special_tokens=True)
        input.to(device)
        
        answer_start, answer_end = (answer_starts[pointer:min(pointer + batch_size, len(contexts))], answer_ends[pointer:min(pointer + batch_size, len(contexts))])
        true_labels1 = torch.LongTensor(np.array(answer_start)).to(device)
        true_labels2 = torch.LongTensor(np.array(answer_end)).to(device)
            
        output = model(**input)
        if output is None: continue
        pred_indicies1 = output['start_logits']
        pred_indicies2 = output['end_logits']
        # print(pred_indicies1.shape, true_labels1.shape)
        loss1 = loss_fn(pred_indicies1,true_labels1)
        loss2 = loss_fn(pred_indicies2,true_labels2)
        loss = loss1 + loss2

        # back propagate
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # update weights 
        optimizer.step()

        # update training rate
        scheduler.step()

        if step_cnt%int(len(contexts)/batch_size/10) == 0 or step_cnt == math.ceil(len(contexts)*1./batch_size):
            acc = evaluate(model,dev_data,device,tokenizer,mute=True)
            print('==> step {} dev acc: {}'.format(step_cnt,acc))
            if acc > best_acc:
                best_acc = acc
                best_weight = copy.deepcopy(model.cpu().state_dict())
                #model.to(device)
                # hsy save
                torch.save(model.cpu().state_dict(),'./best_model.pth')
                model.to(device)
                print('verbose.......model saved at step {}'.format(step_cnt))

    return best_weight
    
def evaluate(model, test_data, device, tokenizer, mute=False, batch_size=10):
    model.eval()
    contexts, questions, answer_starts, answer_ends = test_data
    all_labels = []
    all_predict = np.array([])
    with torch.no_grad():
        for pointer in range(0, len(contexts), batch_size):            
            input = tokenizer(contexts[pointer:min(pointer + batch_size, len(contexts))], questions[pointer:min(pointer + batch_size, len(contexts))], return_tensors="pt",truncation=True, padding=True, return_token_type_ids=True, add_special_tokens=True)
            input.to(device)

            answer_start, answer_end = (answer_starts[pointer:min(pointer + batch_size, len(contexts))], answer_ends[pointer:min(pointer + batch_size, len(contexts))])
            all_labels = all_labels + answer_start 
            all_labels = all_labels + answer_end
            
            outputs = model(**input)
            
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            predict_start = [np.argmax(pp) for pp in start_logits.cpu()]
            predict_end = [np.argmax(pp) for pp in end_logits.cpu()]
            all_predict = np.concatenate((all_predict, predict_start), axis=None)
            all_predict = np.concatenate((all_predict, predict_end), axis=None)
    assert len(all_predict) == len(all_labels)


    acc = len([i for i in range(len(all_labels)) if all_predict[i]==all_labels[i]])*1./len(all_labels)

    if not mute:
        print('==>acc<==', acc)

    return acc

train_d = load_dataset('tydiqa', name = 'primary_task', split = 'train')
train_d[0]

#'finnish': 6855, 'telugu': 5563, 'russian': 6490, 'arabic': 14805, 'indonesian': 5702, 'english': 3696, 'swahili': 2755, 'korean': 1625, 'bengali': 2390}
l = ['english', 'finnish', 'telugu', 'russian', 'arabic', 'indonesian', 'swahili', 'korean', 'bengali']

train_d = load_dataset('tydiqa', name = 'secondary_task', split = 'train')
train_d = train_d.shuffle()
test_d = load_dataset('tydiqa', name = 'secondary_task', split = 'validation')

print('verbose data...original dataset: train len {}, test len {}'.format(len(train_d), len(test_d)))

tokenizer=AutoTokenizer.from_pretrained('xlm-roberta-base')
languages = [(t['id'].split("-")[0]) for t in train_d]

# find target language indicies
indexes_train = [i for i,x in enumerate(languages) if x in l]
indexes_dev = indexes_train[-100:]
indexes_train = indexes_train[:-100]
languages = [(t['id'].split("-")[0]) for t in test_d]
indexes_test = [i for i,x in enumerate(languages) if x in l]

train_data = prepare_inputs(indexes_train, train_d, tokenizer)
dev_data = prepare_inputs(indexes_dev, train_d, tokenizer)
test_data = prepare_inputs(indexes_test, test_d, tokenizer)


print('verbose data...dataset to used: train len {}, dev len {}, test len {}'.format(len(train_data[0]), len(dev_data[0]), len(test_data[0])))

epoch_num = 2
#batch_size = 5
batch_size = 12 #hsy
#batch_size = 128 #hsy
# warmup_percent = 0.2
warmup_percent = 0.1
max_grad_norm = 1 #TODO
scheduler_setting = 'WarmupLinear'
device = 'cuda'
total_steps = math.ceil(epoch_num*len(train_data[0])*1./batch_size)
warmup_steps = int(total_steps*warmup_percent)

model_path = 'ModelWeights'
iter = 0
# for i in range(0, epoch_num):
#     if os.path.exists('./model weights/' + model_path + str(i)):
#       iter = i
identifier = "xlm-roberta-base" if iter == 0 else './model weights/' + model_path + str(iter)
iter = iter + 1 if iter != 0 else iter

print(identifier)
model = AutoModelForQuestionAnswering.from_pretrained('xlm-roberta-base')
tokenizer=AutoTokenizer.from_pretrained('xlm-roberta-base')

# hsy
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#     model = nn.DataParallel(model)

model.to(device)
optimizer = AdamW(model.parameters(),lr=3e-5,eps=1e-6,correct_bias=False)
scheduler = get_scheduler(optimizer, scheduler_setting, warmup_steps=warmup_steps, t_total=total_steps) 

st_time = time.time()
best_weight = None
for i in range(0, epoch_num):
    best_weight = train(model, optimizer, scheduler, train_data, dev_data, batch_size, device, max_grad_norm, tokenizer, best_acc = -1)
    #model.load_state_dict(best_weight)
    # model.save_pretrained(save_directory = './model weights/'+ model_path + str(i))

print('Training done! cost time {} min'.format((time.time()-st_time)//60))
model.load_state_dict(best_weight)
evaluate(model, test_data, device, tokenizer, mute=False, batch_size=10)

