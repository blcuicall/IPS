import requests
import json
import subword_nmt.apply_bpe as apply_bpe
import fastBPE
# bpe = fastBPE.fastBPE(codes_path, vocab_path)
# bpe.apply(['hahahahahahahahaha'])
# path = './tmp/multi_mask_data'
bpe = fastBPE.fastBPE('one_billion_word_strict_1kw/code.bpe')

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


f_pred = open('predict/pred.txt', 'w')
count = 0

with open('one_billion_word_strict_1kw/test_keys.txt', 'r') as f_key:
    print(f_key)
    for keys_line in f_key:
        
        keys_list = keys_line.strip().split(' ')
        keys_list = bpe.apply(keys_list)
        span_list = []
        
        start_span = '[pred]'
        for key in keys_list:
            start_span += ' ' + key + ' [blank]'
       
        for i in range(len(keys_list)+1):
            
            sents_json = json.dumps([start_span])
            res = requests.post(f"http://202.112.194.65:10808/",
                    json={"span_sent":sents_json})
            pred_span = res.json()[0]
            
            pred_span = pred_span.replace('[bos]', '') # remove [bos] [eos]
            pred_span = pred_span.replace('[eos]', '')
            
            start_span = start_span.replace('[pred]', pred_span, 1)
            start_span = start_span.replace('[blank]', '[pred]', 1)
            # 多个空格变成一个空格
            start_span = ' '.join(start_span.split())
        print('prediction:\n', start_span)
        f_pred.write(start_span+'\n')
