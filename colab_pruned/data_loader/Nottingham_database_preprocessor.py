import os
import music21
from music21 import *
import pickle
import json
from copy import copy
from data_loader.Nottingham_database_preprocessor_util import *

#----------------------------------------------------Start of discription----------------------------------------------------#

# Author: Lu Tongyu, June 5th, 2020

# To read midi to defined formats, to save data, to make dictionaries, to write data of our formats to midi
# This file aims convert midi data to lists and the inverse process

# Our format of rhythm pattern strings:
# for explanation (this dictionary not used)
rhythm_symbol_dictionary={'dotted_whole_note':'N6.000',
                          'dotted_whole_rest':'R6.000',
                          'dotted_whole_hold':'H6.000',
                          'whole_note':'N4.000',
                          'whole_rest':'R4.000',
                          'whole_hold':'H4.000',
                          'dotted_half_note':'N3.000',
                          'dotted_half_rest':'R3.000',
                          'dotted_half_hold':'H3.000',
                          'half_note':'N2.000',
                          'half_rest':'R2.000',
                          'half_hold':'H2.000',
                          'dotted_quarter_note':'N1.500',
                          'dotted_quarter_rest':'R1.500',
                          'dotted_quarter_hold':'H1.500',
                          'quarter_note':'N1.000',
                          'quarter_rest':'R1.000',
                          'quarter_hold':'H1.000',
                          'tri_quarter_note':'N0.667',
                          'tri_quarter_rest':'R0.667',
                          'tri_quarter_hold':'H0.667',
                          'quarter_note':'N0.667',
                          'quarter_rest':'R0.667',
                          'quarter_hold':'H0.667'}#and so forth
#'Note with duration T': 'N'+'{:.3f}'.format(note_duration)

#Rhythm Partition examples:
# 2/4: count_per_measure=2, beat_duration=1.0
# 3/4: count_per_measure=3, beat_duration=1.0
# 4/4: count_per_measure=4, beat_duration=1.0
# 5/4: count_per_measure=5, beat_duration=1.0
# 6/4: count_per_measure=6, beat_duration=1.0
# 3/8: count_per_measure=3, beat_duration=0.5
# 6/8: count_per_measure=6, beat_duration=0.5
# so forth

# pitch integers: -3:controller, -2:meter signature, -1:rest, 0-128: pitches

#----------------------------------------------------End of discription----------------------------------------------------#

#Discription: read midi files to such objects:
#Outputs:
#   rhythm_pattern_list_all: [pieces],where every piece:[list_of_strings],
#   rhythm_pattern_duration_all: [float_lists],where every float_list:[floats(time_per_measure)],
#   melody_pitch_list_all: [pieces],where every piece:[list_of_integers],
#   melody_duration_list_all: [pieces],where every piece:[list_of_floats],
#   aligned_chord_list_all: [pieces],where every piece:[list_of_music21_objs],
#   chord_symbol_list_all: [pieces],where every piece:[music21_objs],
#   max_len_piece: integer
def midi2lists(filepath, make_controller=True):
    midi_files = os.listdir(filepath)
    piece_names = []
    rhythm_pattern_list_all = []
    rhythm_pattern_duration_all = []
    melody_pitch_list_all = []
    melody_duration_list_all = []
    aligned_chord_list_all = []

    chord_symbol_list_all = []
    for file in midi_files:
        if '.mid' not in file:
            continue
        mf = midi.MidiFile()
        mf.open(filepath + r'/' + file)
        mf.read()
        stream_tmp = midi.translate.midiFileToStream(mf)
        mf.close()
        print('read file',file)
        piece_names.append(file)

        if len(stream_tmp.parts)==1:
            print('stream_tmp.parts las length 1, PASS!')
            continue
        elif len(stream_tmp.parts)==2:
            stream1_melody = stream_tmp.parts[0]
            stream2_chord = stream_tmp.parts[1]
        else:
            print('stream_tmp.parts las length more than 2, PASS!')
            continue
            
        instru = instrument.partitionByInstrument(stream1_melody)
        if instru:
            stream_part_melody = instrument.partitionByInstrument(stream1_melody).parts[0]
            stream_part_chord = instrument.partitionByInstrument(stream2_chord).parts[0]
        else:
            stream_part_melody = stream1_melody
            stream_part_chord = stream2_chord

        melody_symbol_list,melody_time_stamp_list=rectify_stream_part(stream_part_melody)
        chord_symbol_list,chord_time_stamp_list=rectify_stream_part(stream_part_chord)

        melody_rhythm_pattern_list, melody_rhythm_pattern_durations, melody_pitch_list, melody_duration_list = get_melody(melody_symbol_list,melody_time_stamp_list)
        aligned_chord_list = get_melody_aligned_with_chord(melody_duration_list,chord_symbol_list,chord_time_stamp_list)

        rhythm_pattern_list_all.append(melody_rhythm_pattern_list)
        rhythm_pattern_duration_all.append(melody_rhythm_pattern_durations)
        melody_pitch_list_all.append(melody_pitch_list)
        melody_duration_list_all.append(melody_duration_list)
        aligned_chord_list_all.append(aligned_chord_list)
        chord_symbol_list_all.append(chord_symbol_list)

    # vocab_int2word={1:'<PAD>',2:'<BOS>',3:'<EOS>',4:'<UNK>',5:'<MASK>',6:'<CLS>'}
    max_len_piece=0
    if make_controller and rhythm_pattern_list_all[0][0]!='<BOS>':
        rest_note = note.Rest('')
        rest_note.duration = duration.Duration(0)
        for id_sen in range(len(rhythm_pattern_list_all)):
            #添加曲子开始和结束的标记
            rhythm_pattern_list_all[id_sen].insert(0,'<BOS>')
            rhythm_pattern_duration_all[id_sen].insert(0,0)
            melody_pitch_list_all[id_sen].insert(0,[-3])
            melody_duration_list_all[id_sen].insert(0,[0])

            aligned_chord_list_all[id_sen].insert(0,[rest_note])
            chord_symbol_list_all[id_sen].insert(0,rest_note)

            rhythm_pattern_list_all[id_sen].append('<EOS>')
            rhythm_pattern_duration_all[id_sen].append(0)
            melody_pitch_list_all[id_sen].append([-3])
            melody_duration_list_all[id_sen].append([0])
            aligned_chord_list_all[id_sen].append([rest_note])
            chord_symbol_list_all[id_sen].append(rest_note)          

    for id_sen in range(len(rhythm_pattern_list_all)):
        if len(rhythm_pattern_list_all[id_sen])>max_len_piece:
            max_len_piece=len(rhythm_pattern_list_all[id_sen])
    return rhythm_pattern_list_all,rhythm_pattern_duration_all,melody_pitch_list_all,melody_duration_list_all,aligned_chord_list_all,chord_symbol_list_all,max_len_piece,piece_names


def save_data_lists(data_folder_name,\
    rhythm_pattern_list_all,\
    rhythm_pattern_duration_all,\
    melody_pitch_list_all,\
    melody_duration_list_all,\
    aligned_chord_list_all,\
    chord_symbol_list_all):
    filename1 = data_folder_name+'/rhythm_pattern_list_all.data'
    filename2 = data_folder_name+'/rhythm_pattern_duration_all.data'
    filename3 = data_folder_name+'/melody_pitch_list_all.data'
    filename4 = data_folder_name+'/melody_duration_list_all.data'
    filename5 = data_folder_name+'/aligned_chord_list_all.data'
    filename6 = data_folder_name+'/chord_symbol_list_all.data'
    # 存储变量的文件的名字
    # 以二进制写模式打开目标文件
    f1 = open(filename1, 'wb')
    f2 = open(filename2, 'wb')
    f3 = open(filename3, 'wb')
    f4 = open(filename4, 'wb')
    f5 = open(filename5, 'wb')
    f6 = open(filename6, 'wb')
    # 将变量存储到目标文件中区
    pickle.dump(rhythm_pattern_list_all, f1)
    pickle.dump(rhythm_pattern_duration_all, f2)
    pickle.dump(melody_pitch_list_all, f3)
    pickle.dump(melody_duration_list_all, f4)
    pickle.dump(aligned_chord_list_all, f5)
    pickle.dump(chord_symbol_list_all, f6)
    # 关闭文件
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    f6.close()
    max_len_piece=0
    for id_sen in range(len(rhythm_pattern_list_all)):
        if len(rhythm_pattern_list_all[id_sen])>max_len_piece:
            max_len_piece=len(rhythm_pattern_list_all[id_sen])
    print('sentence_max_length is: ',max_len_piece)
    print('e.g. sentence[0]: ',rhythm_pattern_list_all[0])
    return max_len_piece

def load_data_lists(data_folder_name):
    filename1 = data_folder_name+'/rhythm_pattern_list_all.data'
    filename2 = data_folder_name+'/rhythm_pattern_duration_all.data'
    filename3 = data_folder_name+'/melody_pitch_list_all.data'
    filename4 = data_folder_name+'/melody_duration_list_all.data'
    filename5 = data_folder_name+'/aligned_chord_list_all.data'
    filename6 = data_folder_name+'/chord_symbol_list_all.data'
    # 以二进制读模式打开目标文件
    f1 = open(filename1, 'rb')
    f2 = open(filename2, 'rb')
    f3 = open(filename3, 'rb')
    f4 = open(filename4, 'rb')
    f5 = open(filename5, 'rb')
    f6 = open(filename6, 'rb')
    # 将文件中的变量加载到当前工作区
    rhythm_pattern_list_all = pickle.load(f1)
    rhythm_pattern_duration_all = pickle.load(f2)
    melody_pitch_list_all = pickle.load(f3)
    melody_duration_list_all = pickle.load(f4)
    aligned_chord_list_all = pickle.load(f5)
    chord_symbol_list_all = pickle.load(f6)
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    f6.close()
    max_len_piece=0
    for id_sen in range(len(rhythm_pattern_list_all)):
        if len(rhythm_pattern_list_all[id_sen])>max_len_piece:
            max_len_piece=len(rhythm_pattern_list_all[id_sen])
    return rhythm_pattern_list_all,rhythm_pattern_duration_all,melody_pitch_list_all,melody_duration_list_all,aligned_chord_list_all,chord_symbol_list_all,max_len_piece


def make_dictionary(rhythm_pattern_list_all,bias_tokens_n = 20):
    vocab_int2word={1:'<PAD>',2:'<BOS>',3:'<EOS>',4:'<UNK>',5:'<MASK>',6:'<CLS>'}
    vocab_word2int= dict(zip(vocab_int2word.values(), vocab_int2word.keys()))
    word_num = bias_tokens_n
    for piece in rhythm_pattern_list_all:
        for word in piece:
            if not (word in vocab_int2word.values()):
                #print(word)
                vocab_int2word.update({word_num:word})
                word_num=word_num+1
    vocab_word2int = {value:key for key,value in vocab_int2word.items()}
    return vocab_int2word,vocab_word2int

def save_dictionary_as_json(data_folder_name,vocab_int2word,vocab_word2int):
    filename3 = data_folder_name+'/vocab_int2word.json'
    filename4 = data_folder_name+'/vocab_word2int.json'
    # 存储变量的文件的名字
    # 以二进制写模式打开目标文件
    with open(filename3, 'w') as f3:
    # 将变量存储到目标文件中区
        json.dump(vocab_int2word, f3)
    # 关闭文件
    f3.close()
    with open(filename4, 'w') as f4:
    # 将变量存储到目标文件中区
        json.dump(vocab_word2int, f4)
    # 关闭文件
    f4.close()

def jsonKeys2int(x):
    if isinstance(x, dict):
        return {int(k):v for k,v in x.items()}
    return x

def load_dictionary_as_json(data_folder_name):
    filename3 = data_folder_name+'/vocab_int2word.json'
    filename4 = data_folder_name+'/vocab_word2int.json'
    # 以二进制读模式打开目标文件
    f3 = open(filename3, 'r')
    f4 = open(filename4, 'r')
    # 将文件中的变量加载到当前工作区
    vocab_int2word = json.load(f3, object_hook=jsonKeys2int)
    vocab_word2int = json.load(f4)
    f3.close()
    f4.close()
    return vocab_int2word,vocab_word2int

def result_tuplelist_to_midi(result_list_of_tuples,midi_results_path,max_file_num=None):
    if not max_file_num:
        max_file_num = len(result_list_of_tuples)
    for index,piece in enumerate(result_list_of_tuples):
        pred=piece[0]
        target=piece[1]
        print('Writing midi file number',index)
        test_stream_result,_=translate_rhythm_string_list_into_stream(pred)
        test_stream_target,_=translate_rhythm_string_list_into_stream(target)
        test_stream_result.write('midi', fp=midi_results_path+'/the_test_stream_result'+str(index)+'.mid')
        test_stream_target.write('midi', fp=midi_results_path+'/the_test_stream_target'+str(index)+'.mid')
        if index>max_file_num:
            break

def write_rhythm_patterns_to_midi(rhythm_patterns_lists,midi_results_path='/',file_name='/midi_thythm',max_file_num=None):
    if not max_file_num:
        max_file_num = len(rhythm_patterns_lists)
    for index,piece in enumerate(rhythm_patterns_lists):
        print('Writing midi file number',index)
        test_stream_result,_=translate_rhythm_string_list_into_stream(piece)
        test_stream_result.write('midi', fp=midi_results_path+file_name+str(index)+'.mid')
        if index>max_file_num:
            break

def write_notes_to_midi(notes_lists,duration_lists,aligned_chord_lists=None,rhythm_pattern_lists=None,midi_results_path='/',file_name='/midi_thythm',max_file_num=None):
    if not max_file_num:
        max_file_num = len(notes_lists)
    for i in range(max_file_num):
        notes_list = notes_lists[i]
        duration_list = duration_lists[i]
        if aligned_chord_lists:
            aligned_chord_list = aligned_chord_lists[i]
        else:
            aligned_chord_list = None
        if rhythm_pattern_lists:
            rhythm_pattern_list = rhythm_pattern_lists[i]
        else:
            rhythm_pattern_list = None
        
        note_stream,chord_stream = \
        translate_note_list_into_stream(notes_list,duration_list,aligned_chord_list,rhythm_pattern_list)

        print('Writing midi file (pitch+chord), number ',i)

        midi_score = stream.Score(id='mainScore')
        midi_part0 = stream.Part(id='part0')
        midi_part0.append(note_stream)
        midi_score.insert(0,midi_part0)
        if aligned_chord_lists:
            midi_part1 = stream.Part(id='part1')
            midi_part1.append(chord_stream)
            midi_score.insert(0,midi_part1)

        midi_score.write('midi', fp=midi_results_path+file_name+str(i)+'.mid')

def rhythm_pattern_to_n_note(rhythm_string):
    splitted = rhythm_string.split('|')
    if len(splitted)<=1:
        return 0
    if '' in splitted:
        return 0
    splitted2 = rhythm_string.split(',')

    if rhythm_string[0] == '|':
        return 0
    elif rhythm_string[0] == '<' and '|' not in rhythm_string:
        return 0
    else:
        meter_index=rhythm_string.find('|')
        meter_str=rhythm_string[meter_index+1:]
        tmp_str=rhythm_string.replace('|'+meter_str,'')
        obj_list=tmp_str.split(',')
        n_notes = 0
        for obj in obj_list:
            if '<' in obj:
                continue
            else:
                n_notes += 1
        return n_notes

def pitch2octave(pitch_list):
    if(type(pitch_list)==list):
        list_buff = [-1 for _ in range(len(pitch_list))]
        for i,p in enumerate(pitch_list):
            if p>=0:
                list_buff[i]=max(p//12-1,0)
            else:
                list_buff[i]=p
    else:
        p=pitch_list
        list_buff = -1
        if p>=0:
            list_buff=max(p//12-1,0)
        else:
            list_buff=p
    return list_buff

def pitch2pitchclass(pitch_list):
    if(type(pitch_list)==list):
        list_buff = [-1 for _ in range(len(pitch_list))]
        for i,p in enumerate(pitch_list):
            if p>=0:
                list_buff[i]=p%12
            else:
                list_buff[i]=p
    else:
        p=pitch_list
        list_buff = -1
        if p>=0:
            list_buff=p%12
        else:
            list_buff=p
    return list_buff

def chord_obj_to_string(chord_obj_list):
    list_buff = ['None' for _ in range(len(chord_obj_list))]
    for i,ch in enumerate(chord_obj_list):
        if type(ch)==note.Rest:
            list_buff[i] = 'Rest'
        elif type(ch)==chord.Chord:
            list_buff[i] = ch.commonName
        else:
            list_buff[i] = 'None'
    return list_buff

def chord_obj_to_bass(chord_obj_list):
    list_buff = ['None' for _ in range(len(chord_obj_list))]
    for i,ch in enumerate(chord_obj_list):
        if type(ch)==note.Rest:
            list_buff[i] = -1
        elif type(ch)==chord.Chord:
            list_buff[i] = ch.bass().pitchClass
        else:
            list_buff[i] = -3
    return list_buff

def make_dictionary_for_chord_obj(aligned_chord_list_all, bias_tokens_n = 20):
    vocab_int2word={1:'<PAD>',2:'<BOS>',3:'<EOS>',4:'<UNK>',5:'<MASK>',6:'<CLS>'}
    vocab_word2int= dict(zip(vocab_int2word.values(), vocab_int2word.keys()))
    word_num = bias_tokens_n
    for piece in aligned_chord_list_all:
        for measure in piece:
            chord_str_list = chord_obj_to_string(measure)
            for word in chord_str_list:
                if not (word in vocab_int2word.values()):
                    #print(word)
                    vocab_int2word.update({word_num:word})
                    word_num=word_num+1
    vocab_word2int = {value:key for key,value in vocab_int2word.items()}
    return vocab_int2word,vocab_word2int

# def chord_obj_list_all_to_str(aligned_chord_list_all):
#     aligned_chord_type_str_list_all = aligned_chord_list_all
#     for p,piece in enumerate(aligned_chord_list_all):
#         for i,measure in enumerate(piece):
#             chord_str_list = chord_obj_to_string(measure)
#             assert len(chord_bass_list)==len(chord_str_list), 'Length error: root and chord type do not align'
#             aligned_chord_type_str_list_all[p][i] = chord_str_list
#     return aligned_chord_type_str_list_all

def make_dictionary_for_chord_str(aligned_chord_type_str_list_all, bias_tokens_n = 20):
    vocab_int2word={1:'<PAD>',2:'<BOS>',3:'<EOS>',4:'<UNK>',5:'<MASK>',6:'<CLS>'}
    vocab_word2int= dict(zip(vocab_int2word.values(), vocab_int2word.keys()))
    word_num = bias_tokens_n
    for piece in aligned_chord_type_str_list_all:
        for measure in piece:
            for word in measure:
                if not (word in vocab_int2word.values()):
                    #print(word)
                    vocab_int2word.update({word_num:word})
                    word_num=word_num+1
    vocab_word2int = {value:key for key,value in vocab_int2word.items()}
    return vocab_int2word,vocab_word2int


def pitch2onehot(pitch_num,PITCH_DIM_INDEX=127,REST_DIM_INDEX=128,ERROR_DIM_INDEX=129):
    #pitch dims: 0-127
    #rest dim: 128
    #error dim: 129
    pitch_num = int(pitch_num)
    one_hot_pitch = [0 for _ in range(ERROR_DIM_INDEX+1)]
    if pitch_num>=0 and pitch_num<=PITCH_DIM_INDEX:
        one_hot_pitch[pitch_num] = 1
    elif pitch_num==-1:
        one_hot_pitch[REST_DIM_INDEX] = 1
    elif pitch_num<=-2:
        one_hot_pitch[ERROR_DIM_INDEX] = 1
    else:
        print('Invalid pitch input!')
    return one_hot_pitch

def octave2onehot(octave_num,OCTAVE_DIM_INDEX=9,REST_DIM_INDEX=10,ERROR_DIM_INDEX=11):
    #octave dims: 0-9
    #rest dim: 10
    #error dim: 11
    octave_num = int(octave_num)
    one_hot_octave = [0 for _ in range(ERROR_DIM_INDEX+1)]
    if octave_num>=0 and octave_num<=OCTAVE_DIM_INDEX:
        one_hot_octave[octave_num] = 1
    elif octave_num==-1:
        one_hot_octave[REST_DIM_INDEX] = 1
    elif octave_num<=-2:
        one_hot_octave[ERROR_DIM_INDEX] = 1
    else:
        print('Invalid octave input!')
    return one_hot_octave

def pitchclass2onehot(pitchclass,CLASS_DIM_INDEX=11,REST_DIM_INDEX=12,ERROR_DIM_INDEX=13):
    #pitchclass dims: 0-11
    #rest dim: 12
    #error dim: 13
    pitchclass = int(pitchclass)
    one_hot_pitchclass = [0 for _ in range(ERROR_DIM_INDEX+1)]
    if pitchclass>=0 and pitchclass<=CLASS_DIM_INDEX:
        one_hot_pitchclass[pitchclass] = 1
    elif pitchclass==-1:
        one_hot_pitchclass[REST_DIM_INDEX] = 1
    elif pitchclass<=-2:
        one_hot_pitchclass[ERROR_DIM_INDEX] = 1
    else:
        print('Invalid pitchclass input!')
    return one_hot_pitchclass

def idx2onehot(idx,vocab_size):
    idx= int(idx)
    one_hot = [0 for _ in range(vocab_size)]
    if idx>=0 and idx<=vocab_size:
        one_hot[idx] = 1
    else:
        print('Invalid idx input!')
    return one_hot

def rhythmword2onehot(rhythmword,vocab_word2idx):
    vocab_size = len(vocab_word2idx)
    rhythmidx = vocab_word2idx[rhythmword]
    return  idx2onehot(rhythmidx,vocab_size)

def rhythm_idx_to_n_note(rhythm_idx, vocab_int2word):
    rhythm_string = vocab_int2word[rhythm_idx]
    n_note= rhythm_pattern_to_n_note(rhythm_string)
    return n_note

def chord_list_all_to_idx(aligned_chord_type_str_all,vocab_word2int):
    aligned_chord_type_idx_all = \
    [[[vocab_word2int[word] \
    for j,word in enumerate(measure)] \
    for i,measure in enumerate(piece)] \
    for p,piece in enumerate(aligned_chord_type_str_all)]
    # for p,piece in enumerate(aligned_chord_type_str_all):
    #     for i,measure in enumerate(piece):
    #         for j,word in enumerate(measure):
    #             aligned_chord_type_idx_all[p][i][j]=vocab_word2int[word]
    return aligned_chord_type_idx_all

class Nottingham_dataloader():
    def __init__(self,data_folder_name):
        self.data_folder_name = data_folder_name

        self.piece_names = []
        self.rhythm_pattern_list_all = []
        self.rhythm_pattern_duration_all = []
        self.melody_pitch_list_all = [] #-3:controller, -2:meter signature, -1:rest
        self.melody_duration_list_all = []
        self.aligned_chord_list_all = []
        self.aligned_chord_type_all = []
        self.aligned_chord_type_idx_all = []
        self.aligned_chord_bass_all = []
        self.aligned_chord_func_all = []

        self.max_len_piece = 0
        self.vocab_int2word={}
        self.vocab_word2int={}
        self.vocab_int2word_chord={}
        self.vocab_word2int_chord={}

    def midi2lists(self,filepath, make_controller=True):
        self.rhythm_pattern_list_all, \
        self.rhythm_pattern_duration_all, \
        self.melody_pitch_list_all, \
        self.melody_duration_list_all, \
        self.aligned_chord_list_all, \
        self.chord_symbol_list_all, \
        self.max_len_piece, self.piece_names = midi2lists(filepath, make_controller=True)
        print('data read successfully from midi files')

    def get_chord_types(self):
        print('converting chord types')
        n_piece = len(self.aligned_chord_list_all)
        self.aligned_chord_type_all = [[[] for _ in range(len(self.aligned_chord_list_all[i]))] for i in range(n_piece)]
        for i,piece in enumerate(self.aligned_chord_list_all):
            print("\r processing piece number ",i,end="",flush=True)
            for j,measure in enumerate(piece):
                self.aligned_chord_type_all[i][j]=chord_obj_to_string(measure)
        print('successfully converted chord types')

    def get_chord_bass(self):
        print('converting chord bass')
        n_piece = len(self.aligned_chord_list_all)
        self.aligned_chord_bass_all = [[[] for _ in range(len(self.aligned_chord_list_all[i]))] for i in range(n_piece)]
        for i,piece in enumerate(self.aligned_chord_list_all):
            print("\r processing piece number ",i,end="",flush=True)
            for j,measure in enumerate(piece):
                self.aligned_chord_bass_all[i][j]=chord_obj_to_bass(measure)
        print('successfully converted chord bass')


    def save_data_lists(self):
        save_data_lists(self.data_folder_name, \
                        self.rhythm_pattern_list_all, \
                        self.rhythm_pattern_duration_all, \
                        self.melody_pitch_list_all, \
                        self.melody_duration_list_all, \
                        self.aligned_chord_list_all, \
                        self.chord_symbol_list_all)
        filename = self.data_folder_name+'/piece_names.data'
        # 以二进制读模式打开目标文件
        f = open(filename, 'wb')
        # 将文件中的变量加载到当前工作区
        pickle.dump(self.piece_names, f)
        f.close()
        print('data saved successfully to '+self.data_folder_name)
 
    def save_data_lists_split_train_and_valid(self,proportion=0.6):
        train_folder_name = self.data_folder_name+"/train_data"
        valid_folder_name = self.data_folder_name+"/valid_data"
        n_pieces = len(self.rhythm_pattern_list_all)
        n_train = int(n_pieces*proportion)
        save_data_lists(train_folder_name, \
                        self.rhythm_pattern_list_all[0:n_train], \
                        self.rhythm_pattern_duration_all[0:n_train], \
                        self.melody_pitch_list_all[0:n_train], \
                        self.melody_duration_list_all[0:n_train], \
                        self.aligned_chord_list_all[0:n_train], \
                        self.chord_symbol_list_all[0:n_train])
        filename = train_folder_name+'/piece_names.data'
        # 以二进制读模式打开目标文件
        f = open(filename, 'wb')
        # 将文件中的变量加载到当前工作区
        pickle.dump(self.piece_names[0:n_train], f)
        f.close()
        save_data_lists(train_folder_name, \
                        self.rhythm_pattern_list_all[n_train:], \
                        self.rhythm_pattern_duration_all[n_train:], \
                        self.melody_pitch_list_all[n_train:], \
                        self.melody_duration_list_all[n_train:], \
                        self.aligned_chord_list_all[n_train:], \
                        self.chord_symbol_list_all[n_train:])
        filename = valid_folder_name+'/piece_names.data'
        # 以二进制读模式打开目标文件
        f = open(filename, 'wb')
        # 将文件中的变量加载到当前工作区
        pickle.dump(self.piece_names[n_train:], f)
        f.close()
        print('data saved successfully to '+self.data_folder_name)

    def load_data_lists(self):
        print('loading data from'+self.data_folder_name)
        self.rhythm_pattern_list_all, \
        self.rhythm_pattern_duration_all, \
        self.melody_pitch_list_all, \
        self.melody_duration_list_all, \
        self.aligned_chord_list_all, \
        self.chord_symbol_list_all, \
        self.max_len_piece = load_data_lists(self.data_folder_name)
        filename = self.data_folder_name+'/piece_names.data'
        # 以二进制读模式打开目标文件
        f = open(filename, 'rb')
        # 将文件中的变量加载到当前工作区
        self.piece_names = pickle.load(f)
        f.close()
        print('data loaded successfully from'+self.data_folder_name)

    def make_dictionary(self):
        self.vocab_int2word,self.vocab_word2int = make_dictionary(self.rhythm_pattern_list_all)
        print('dictionary generated successfully')

    def make_dictionary_for_chord_obj(self):
        self.vocab_int2word_chord,self.vocab_word2int_chord = make_dictionary_for_chord_str(self.aligned_chord_type_all)
        print('chord dictionary generated successfully')

    def chord_list_all_to_idx(self):
        self.aligned_chord_type_idx_all = chord_list_all_to_idx(self.aligned_chord_type_all,self.vocab_word2int_chord)
        print('chord type idx generated successfully')

    def save_rhythm_dictionary_as_json(self, child_folder=''):
        this_folder_name = self.data_folder_name+child_folder
        save_dictionary_as_json(this_folder_name,self.vocab_int2word,self.vocab_word2int)
        print('dictionary saved successfully to '+this_folder_name )

    def save_chord_dictionary_as_json(self, child_folder=''):
        this_folder_name = self.data_folder_name+child_folder
        save_dictionary_as_json(this_folder_name,self.vocab_int2word_chord,self.vocab_word2int_chord)
        print('dictionary saved successfully to '+this_folder_name )

    def load_dictionary_as_json(self, child_folder=''):
        this_folder_name = self.data_folder_name+child_folder
        print('loading dictionary from'+this_folder_name)
        vocab_int2word, vocab_word2int = load_dictionary_as_json(this_folder_name)
        print('dictionary loaded successfully from'+this_folder_name )
        return vocab_int2word, vocab_word2int

    def translate_rhythm_string_list_into_stream(self,string_list,place_for_controller=True,DURATION_EPS_FOR_RECONSTR = 0.01):
        symbol_stream,duration_list = \
        translate_rhythm_string_list_into_stream(string_list,place_for_controller=True,DURATION_EPS_FOR_RECONSTR = 0.01)
        return symbol_stream,duration_list
        
    def translate_note_list_into_stream(self,note_measure_list,note_durations_list,chord_measure_list=None,rhythm_list=None,DURATION_EPS_FOR_RECONSTR = 0.01):
        note_stream,chord_stream = \
        translate_note_list_into_stream(note_measure_list,note_durations_list,chord_measure_list=None,rhythm_list=None1)
        return note_stream,chord_stream

    def result_tuplelist_to_midi(self,result_list_of_tuples,midi_results_path='/',max_file_num=None):
        result_tuplelist_to_midi(result_list_of_tuples,midi_results_path,max_file_num)
        print('rhythm list tuple successfully saved to midi at'+midi_results_path)

    def write_rhythm_patterns_to_midi(self,rhythm_patterns_lists,midi_results_path='/',file_name='/midi_thythm',max_file_num=None):
        write_rhythm_patterns_to_midi(self,rhythm_patterns_lists,midi_results_path,file_name,max_file_num)
        print('rhythm list successfully saved to midi at'+midi_results_path)

    def write_notes_to_midi(self,notes_lists,duration_lists,aligned_chord_lists=None,rhythm_pattern_lists=None,midi_results_path='/',file_name='/midi_piece',max_file_num=None):
        write_notes_to_midi(notes_lists,duration_lists,aligned_chord_lists,rhythm_pattern_lists,midi_results_path,file_name,max_file_num)
        print('Successfully saved to midi at'+midi_results_path)


class Measurewise_dataloader():
    def __init__(self,nottingham_dataloader):
        self.n_piece = len(nottingham_dataloader.rhythm_pattern_list_all)
        self.measures = []
        for p in range(self.n_piece):
            for i in range(len(nottingham_dataloader.rhythm_pattern_list_all[p])):
                measure_rhythm = nottingham_dataloader.rhythm_pattern_list_all[p][i]
                measure_pitches = nottingham_dataloader.melody_pitch_list_all[p][i]
                measure_durations = nottingham_dataloader.melody_duration_list_all[p][i]
                measure_chordidx = nottingham_dataloader.aligned_chord_type_idx_all[p][i]
                measure_chordbass = nottingham_dataloader.aligned_chord_bass_all[p][i]
                measure_package = (measure_rhythm,measure_pitches,measure_durations,measure_chordidx,measure_chordbass)
                self.measures.append(measure_package)
        self.length = len(self.measures)
        self.vocab_int2word_rhythm=nottingham_dataloader.vocab_int2word
        self.vocab_word2int_rhythm=nottingham_dataloader.vocab_word2int
        self.vocab_int2word_chord=nottingham_dataloader.vocab_int2word_chord
        self.vocab_word2int_chord=nottingham_dataloader.vocab_word2int_chord

    def __len__(self):
        return self.length

    def get_measure_one_hot(self,index):
        (measure_rhythm,measure_pitches,measure_durations,measure_chordidx,measure_chordbass) = self.measures[index]
        rhythm_onehot = rhythmword2onehot(measure_rhythm,self.vocab_word2int_rhythm)
        pitch_onehots = [None for n in range(len(measure_pitches))]
        chord_onehots = [None for n in range(len(measure_chordidx))]
        bass_onehots = [None for n in range(len(measure_chordbass))]
        for n in range(len(measure_pitches)):
            pitch_onehots[n] = pitch2onehot(measure_pitches[n])
            chord_onehots[n] = idx2onehot(measure_chordidx[n],len(self.vocab_int2word_chord))
            bass_onehots[n] = pitchclass2onehot(measure_chordbass[n])
        measure_package = (rhythm_onehot,pitch_onehots,measure_durations,chord_onehots,bass_onehots)
        return measure_package