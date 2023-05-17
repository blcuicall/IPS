import requests
import json
import subword_nmt.apply_bpe as apply_bpe
import fastBPE
# bpe = fastBPE.fastBPE(codes_path, vocab_path)
# bpe.apply(['hahahahahahahahaha'])
# path = './tmp/multi_mask_data'
bpe = fastBPE.fastBPE('News/multi_mask_data/code.bpe')

# sents = ['[pred] Don@@ o@@ gh@@ ue [blank] neatly [blank] drama [blank] page [blank] ',
#          'Some may have to pay [pred] cent [blank] ', 
#          'Some may have to pay up to 50 per cent [pred] ',
#          'It cost the church $ 36 million [pred] hold [blank] pilgrims [blank]']
# sents_json = json.dumps(sents)

# res = requests.post(f"http://202.112.194.65:10808/",
#                     json={"span_sent":sents_json})
# print(res.json())
# res = res.json()
# print(type(res))


f_pred = open('predict/Yelp/pos/batch_decode_beam5_len2_pred.txt', 'w')
count = 0

input_set = {}
with open('Yelp/pos/multi_mask_data/test_keys.txt', 'r') as f_key:
    for keys_line in f_key:
        
        keys_list = keys_line.strip().split(' ')
        keys_list = bpe.apply(keys_list)
        span_list = []
        
        start_span = '[pred]'
        for key in keys_list:
            start_span += ' ' + key + ' [blank]'
        #input_set.append(start_span)
        input_set[count] = start_span
        count += 1


sents = {}
batch_size = 512
# for i, stage1_sent in enumerate(input_set):
def decode_batch(index_list, input_list, max_keys_num=4):
    res_sents = {}
    for _ in range(max_keys_num+1):
        sents_json = json.dumps(input_list)
        tmp_input_list = input_list
        res = requests.post(f"http://202.112.194.65:10809/", json={"span_sent":sents_json})
        pred_spans = res.json()
        input_list = []
        for i, input_sent,  seg in zip(index_list, tmp_input_list, pred_spans):
            seg = seg.replace('[bos]', '')
            seg = seg.replace('[eos]', '')
            input_sent = input_sent.replace('[pred]', seg, 1)
            input_sent = ' '.join(input_sent.split())
            if '[blank]' in input_sent:
                input_sent = input_sent.replace('[blank]', '[pred]', 1)
            else:
                res_sents[i] = input_sent
            input_list.append(input_sent)
    return res_sents

num = len(input_set)
s = 0
# for i in range(1, num+1):
#     print(input_set[i])

c = 0
print(num)
while s<num:
    t = min(s + batch_size, num)
    ind_list = []
    batch_set = []
    for ind in range(s, t):
        ind_list.append(ind)
        batch_set.append(input_set[ind])
    s = t
    print(len(batch_set))
    sents = decode_batch(ind_list, batch_set)
    print(sents)
    for i, sent in zip(ind_list, sents):
        start_span =sents[i]
        tmp = start_span.split('@@ ')
        start_span = ''.join(tmp)
        f_pred.write(start_span +'\n')


# with open('News/multi_mask_data/test_keys.txt', 'r') as f_key:
#     print(f_key)
#     for keys_line in f_key:
        
#         keys_list = keys_line.strip().split(' ')
#         keys_list = bpe.apply(keys_list)
#         span_list = []
        
#         start_span = '[pred]'
#         for key in keys_list:
#             start_span += ' ' + key + ' [blank]'
       
#         for i in range(len(keys_list)+1):
            
#             sents_json = json.dumps([start_span])
#             res = requests.post(f"http://202.112.194.65:10808/",
#                     json={"span_sent":sents_json})
#             pred_span = res.json()[0]
            
#             pred_span = pred_span.replace('[bos]', '') # remove [bos] [eos]
#             pred_span = pred_span.replace('[eos]', '')
            
#             start_span = start_span.replace('[pred]', pred_span, 1)
#             start_span = start_span.replace('[blank]', '[pred]', 1)
#             # 多个空格变成一个空格
#             start_span = ' '.join(start_span.split())
#         count += 1
#         print(count)
#         print('prediction:\n', start_span)
#         f_pred.write(start_span+'\n')
