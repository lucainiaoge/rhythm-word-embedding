import music21
from music21 import *

# Author: Lu Tongyu, June 5th, 2020
# - Change music21 streams to part of chords and melody
# - Change melody into rhythmic lists and corresponding list of notes
# - Change rhythmic lists into rhythmic pattern strings (partitioned by measures, so do the list of melody notes and chords. in this scheme, we did not consider rhythm of chords. So, the size of chord list per measure is the same as that of melody)

#lib: from music21 import *
def type_is_note_or_chord(obj_type):
    if obj_type==note.Note or obj_type==chord.Chord:
        return True
    else:
        return False

def type_is_note_or_chord_or_rest(obj_type):
    if obj_type==note.Note or obj_type==chord.Chord or obj_type==note.Rest:
        return True
    else:
        return False

# input type: [float/<cfractions.Fraction>]*L (len>=1)
# output type: [float/<cfractions.Fraction>]*L (len>=1)
def time2duration(time_stamp_list,last=None):
    length=len(time_stamp_list)
    duration_list = [None]*length
    for i in range(length-1):
        duration_list[i]=time_stamp_list[i+1]-time_stamp_list[i]
    duration_list[length-1]=last
    return duration_list

#Input type: <music21.stream.Part>
#Previous operation: read midi, convert to stream, get main instrument
#lib: from music21 import *
#Output type: [<music21.chord.Chord/<music21.note.Rest>/<music21.meter.TimeSignature>],[float/<cfractions.Fraction>]
#Discription: converge a instrumental part into accessible formation of list
def rectify_stream_part(stream_part,DURATION_EPS = 0.000001):
    symbol_list=[]#用于记录事件符号。如果是音符或者和弦，一律用和弦音升序排列的和弦obj来记载；如果是空拍，就按空拍obj来记载；如果是拍子转换，就用拍号的obj来记载
    time_stamp_list=[]#用于记录每个事件符号的起始时间
    L=len(stream_part)
    current_chord = chord.Chord([])
    
    for i in range(L):
        if i>=len(stream_part):
            break
        event=stream_part[i]
        event_time=event.offset
        # assert boolean_expression, 'assert if boolean_expression==False'
        assert len(time_stamp_list)==len(symbol_list), 'Pitch list not same length as rhythm!'
        type_of_event=type(event)
        
        if type_of_event==meter.TimeSignature:
            symbol_list.append(event)
            time_stamp_list.append(event_time)
            
        elif type_is_note_or_chord(type_of_event):
            current_chord.add(event)
            if i<len(stream_part)-1:
                next_event = stream_part[i+1]
                next_event_time = next_event.offset
                if type_is_note_or_chord(type(next_event)):#如果下一个事件还是一个音符或者和弦
                    if next_event_time-event_time<=DURATION_EPS:#如果是同时发生
                        continue #那么不管，读下一个event
                    else:#如果下一个音符或和弦发生在下一时刻，那么就记载这个和弦
                        symbol_list.append(current_chord)
                        time_stamp_list.append(event_time)
                        current_chord = chord.Chord([])
                elif type(next_event)==note.Rest:
                    if next_event_time-event_time<=DURATION_EPS: #如果音符和休止符同时发生
                        stream_part.pop(i+1)
                        continue #那么把这个没有意义的休止符扔掉，读下一个event
                else: #如果下一个事件既不是音符又不是和弦也不是休止符，那么就记载这个和弦
                    symbol_list.append(current_chord)
                    time_stamp_list.append(event_time)
                    current_chord = chord.Chord([])
            else:#如果读到的最后一个音，那么就记载这个和弦
                symbol_list.append(current_chord)
                time_stamp_list.append(event_time)
                current_chord = chord.Chord([])
        elif type_of_event==note.Rest:
            if i<len(stream_part)-1:
                next_event = stream_part[i+1]
                next_event_time = next_event.offset
                if type_is_note_or_chord(type(next_event)):#如果下一个事件是一个音符或者和弦
                    if next_event_time-event_time<=DURATION_EPS:#如果是同时发生
                        continue #那么忽略掉此时这个休止符，读下一个event
                    else:#如果下一个音符或和弦发生在下一时刻，那么就记载这个休止符
                        symbol_list.append(event)
                        time_stamp_list.append(event_time)
                elif type(next_event)==note.Rest:
                    if next_event_time-event_time<=DURATION_EPS: #如果多个休止符同时发生
                        stream_part.pop(i+1)
                        add=2
                        next_event = stream_part[i+add]
                        next_event_time = next_event.offset
                        while(next_event_time-event_time<=DURATION_EPS):
                            stream_part.pop(i+add)
                            add=add+1
                        continue #那么把下一个没有意义的休止符扔掉（假设连续的休止符是bug情况），读下一个event
                else: #如果下一个事件既不是音符又不是和弦也不是休止符，那么就记载这个休止符
                    symbol_list.append(event)
                    time_stamp_list.append(event_time)
            else:#如果读到的最后一个音，那么就记载这个休止符
                symbol_list.append(event)
                time_stamp_list.append(event_time)
    
    return symbol_list,time_stamp_list


#format of rhythm pattern string: 'NoteOrRest,...,NoteOrRest|N/M'
#e.g. a rhythm pattern of 4/4 meter: 'H1.500,R0.500,N1.000,N1.000|4/4'
#e.g. changing the meter to 3/4: '|3/4'

#Input type: output from rectify_stream_part(stream_part)
#Previous operation: symbol_list,time_stamp_list=rectify_stream_part(stream_part)
#lib: from music21 import *
#Output type:
#   rhythm_pattern_list: [string],
#   rhythm_pattern_durations: [float],
#   melody_pitch_list: [list of integers],
#   melody_duration_list: [list of floats],
#Discription: converge a rectified list of chords into rhythm pattern and notes
def get_melody(symbol_list,time_list,is_duration=False,DURATION_EPS = 0.000001):
    count_per_measure=0 #每个小节数几拍，表示拍号的分子
    element_duration=0 #每拍多少时值
    duration_per_measure=0 #count_per_measure*element_duration
    duration_buffer=0 #现在已经数了多少时值
    
    meter_str=''
    
    #将时间点列表转换为时间段列表（如果本来输入的就是时间段列表那不用管）
    if not is_duration:
        if type_is_note_or_chord_or_rest(type(symbol_list[-1])):
            last_duration = symbol_list[-1].duration.quarterLength
        else:
            last_duration = 0
        duration_list = time2duration(time_list,last_duration)
    else:
        duration_list = time_list

    rhythm_pattern_list = []
    this_pattern = ''
    this_pattern_type = ''#N,R,H
    measure_complete_flag = False #记录是否读完一个小节
    split_flag = False #是否分裂到了两个小节
    rhythm_pattern_durations = []

    melody_pitch_list = [] #list of list
    melody_list_temp = []#list, -1表示空拍，-2表示其他控制符
    melody_duration_list = []# list of list
    melody_duration_temp = []# list

    last_pitch = None #用于比对这次的音和上一个小节的音是否保持，如果保持的话，用'H'去标记，否则用'N'
    this_pitch = None
    
    assert len(time_list)==len(symbol_list), 'Pitch list not same length as rhythm!'
    L=len(symbol_list)
    
    index=0
    while(index<len(symbol_list)):
        symbol=symbol_list[index]
        type_of_symbol=type(symbol)
        if type_of_symbol==meter.TimeSignature:
            count_per_measure=symbol.numerator
            #beat_duration=symbol.beatDuration.quarterLength
            meter_str=symbol.ratioString
            divide_index=meter_str.find('/')
            element_duration=round(4/float(meter_str[divide_index+1:]),3)
            duration_per_measure = count_per_measure*element_duration
            
            print(meter_str,duration_per_measure)
            duration_buffer=0#计数器清零，重新计时
            
            this_pitch=-2

            measure_complete_flag=False
            rhythm_pattern_list.append('|'+meter_str)
            rhythm_pattern_durations.append(round(0,3))
            melody_pitch_list.append([this_pitch])
            melody_duration_list.append([round(0,3)])

            melody_list_temp = []
            melody_duration_temp = []
            
        elif type_is_note_or_chord_or_rest(type_of_symbol):
            this_duration = duration_list[index]
            #print('this_duration',this_duration)
            duration_buffer = duration_buffer+this_duration
            
            if type_of_symbol==chord.Chord:
                this_pitch=symbol.pitches[-1].midi
                if this_pitch==last_pitch and split_flag:
                    this_pattern_type = 'H'
                else:
                    this_pattern_type = 'N'
            elif type_of_symbol==note.Note:
                this_pitch=symbol.pitch.midi
                if this_pitch==last_pitch and split_flag:
                    this_pattern_type = 'H'
                else:
                    this_pattern_type = 'N'
            elif type_of_symbol==note.Rest:
                this_pitch=-1
                this_pattern_type = 'R'

            last_pitch = this_pitch

            if duration_buffer-duration_per_measure > DURATION_EPS:
            #如果加入了这个和弦或者休止符以后，时值长度大于小节总长了的话
                overflow_duration = duration_buffer-duration_per_measure #计算多出来的时值
                left_duration = this_duration-overflow_duration #计算小节内剩下的时值
                if type_is_note_or_chord(type_of_symbol):
                    split_note_left = chord.Chord(symbol)
                    split_note_left.duration = duration.Duration(left_duration)
                    split_note_right = chord.Chord(symbol)
                    split_note_right.duration = duration.Duration(overflow_duration)
                else:
                    split_note_left = note.Rest(symbol)
                    split_note_left.duration = duration.Duration(left_duration)
                    split_note_right = note.Rest(symbol)
                    split_note_right.duration = duration.Duration(overflow_duration)
                #然后将这个和弦或者休止符分裂
                symbol_list[index]=split_note_left
                symbol_list.insert(index+1,split_note_right)
                duration_list[index]=left_duration
                duration_list.insert(index+1,overflow_duration)
                symbol=symbol_list[index]#重新取此时的符号
                
                split_flag = True
                measure_complete_flag=True
                duration_str_to_add = this_pattern_type+'{:.3f}'.format(float(left_duration))
                
            elif duration_buffer-duration_per_measure < -DURATION_EPS:
            #如果没有到小节时值，就把这个和弦或者休止符加进来，然后记载这一次的音符音高（以便与下一次比对，如果一样的话，就用hold）
                #print('duration_per_measure=',duration_per_measure)
                #print('duration_buffer=',duration_buffer)
                split_flag = False
                measure_complete_flag=False
                duration_str_to_add = this_pattern_type+'{:.3f}'.format(float(this_duration))
            
            else:
            #如果恰好读到小节结束
                split_flag = False
                measure_complete_flag=True
                duration_str_to_add = this_pattern_type+'{:.3f}'.format(float(this_duration))
            
            if measure_complete_flag:
                this_pattern = this_pattern+duration_str_to_add+'|'+meter_str
                melody_list_temp.append(this_pitch)
                if split_flag:
                    melody_duration_temp.append(round(left_duration,3))
                else:
                    melody_duration_temp.append(round(this_duration,3))
                
                rhythm_pattern_list.append(this_pattern)
                rhythm_pattern_durations.append(round(duration_buffer,3)) #精确到三位小数
                melody_pitch_list.append(melody_list_temp)
                melody_duration_list.append(melody_duration_temp)

                this_pattern=''
                duration_buffer = 0
                melody_list_temp = []
                melody_duration_temp = []

            else:
                this_pattern = this_pattern+duration_str_to_add+','
                melody_list_temp.append(this_pitch)
                if split_flag:
                    melody_duration_temp.append(round(left_duration,3))
                else:
                    melody_duration_temp.append(round(this_duration,3))

        index = index+1 #该干的干完了之后，index往后推一位
        
    assert len(rhythm_pattern_list)==len(rhythm_pattern_durations), 'Rhythm Pattern list not same length!'
    return rhythm_pattern_list, rhythm_pattern_durations, melody_pitch_list, melody_duration_list


#Discription: align the chord sequence to every notes (along with dufferent measures)
#Input type:
#   melody_duration_list: [list of floats],
#   chord_symbol_list: [music21 objects],
#   chord_time_list: [float] (same length as chord_symbol_list),
#   
#Previous operation: get melody_duration_list through get_melody(symbol_list,time_list)
#
#Output type:
#   aligned_chord_list: [list of music21 objects](the chord alignment in the same piece with melody_duration_list)
#lib: from music21 import *
def get_melody_aligned_with_chord(melody_duration_list,chord_symbol_list,chord_time_list,DURATION_EPS = 0.000001):
    time_stamp = 0
    chord_time_pointer = 0
    length_chord = len(chord_time_list)
    this_chord = []
    #last_chord = []
    this_measure_chord = []
    aligned_chord_list = []
    end_flag = False
    for measure in melody_duration_list:
        for note_dur in measure:
            time_this_chord = chord_time_list[chord_time_pointer] #起始时间
            if chord_time_pointer<length_chord-1:
                time_next_chord = chord_time_list[chord_time_pointer+1]
            else:
                time_next_chord = time_this_chord+100000 #ending
                end_flag = True
            
            if time_stamp>=time_this_chord and time_stamp<time_next_chord:
                this_chord = chord_symbol_list[chord_time_pointer]
            elif time_stamp>=time_next_chord:
                while time_stamp>=time_next_chord:#如果出现一个旋律对应很多和弦
                    time_this_chord=time_next_chord
                    if chord_time_pointer<length_chord-1:
                        chord_time_pointer = chord_time_pointer+1
                        this_chord = chord_symbol_list[chord_time_pointer]
                        if chord_time_pointer<length_chord-1:
                            time_next_chord = chord_time_list[chord_time_pointer+1]
                        else:
                            time_next_chord = time_this_chord+100000 #ending
                            end_flag = True
                    else:
                        this_chord = chord_symbol_list[length_chord-1]
                        break
                    if time_stamp>=time_this_chord and time_stamp<time_next_chord:
                        break
            else:
                this_chord = note.Rest(note_dur)
            
            this_measure_chord.append(this_chord)
            time_stamp = time_stamp+note_dur
            last_chord = this_chord

        aligned_chord_list.append(this_measure_chord)
        this_measure_chord=[]

    return aligned_chord_list


#Discription: the elementwise inverse process of get_melody() for rhythm, with byproduct of duration list
#Input type:
#   rhythm_string: string(format: rhythm pattern),
#   place_for_controller: bool(if True, then generate Rests with duration 0 for controller strings),
#Output type:
#   symbol_list: [music21 objects](corresponding to rhythm_string),
#   duration_list: [floats](corresponding to rhythm_string)
#lib: from music21 import *
def rhythm_pattern_string_to_duration_list(rhythm_string,place_for_controller=True,DURATION_EPS_FOR_RECONSTR = 0.01):
    duration_list=[]
    symbol_list=[]
    if rhythm_string[0] == '|':
        symbol_list.append(meter.TimeSignature(rhythm_string[1:]))
        duration_list.append(0)
    elif rhythm_string[0] == '<' and '|' not in rhythm_string:
        if place_for_controller:
            duration_list.append(0)
            this_obj = note.Rest('')
            this_obj.duration = duration.Duration(0)
            symbol_list.append(this_obj)
        else:
            pass
    else:
        meter_index=rhythm_string.find('|')
        meter_str=rhythm_string[meter_index+1:]

        divide_index=meter_str.find('/')
        element_duration=round(4/float(meter_str[divide_index+1:]),3)
        count_per_measure=int(meter_str[0:divide_index])
        string_duration=count_per_measure*element_duration
        #print('string_duration=',string_duration)

        tmp_str=rhythm_string.replace('|'+meter_str,'')
        obj_list=tmp_str.split(',')
        duration_buff=0

        for obj_str in obj_list:
            obj_type = obj_str[0]
            if obj_type=='H' or obj_type=='N':
                this_obj = note.Note('C5')
                this_duration = float(obj_str[1:])
            elif obj_type=='R':
                this_obj = note.Rest('')
                this_duration = float(obj_str[1:])
            elif obj_type=='<':
                if place_for_controller:
                    this_obj = note.Rest('')
                    this_duration = 0
                else:
                    continue
            if abs(duration_buff+this_duration-string_duration)<=DURATION_EPS_FOR_RECONSTR:
                this_duration = string_duration-duration_buff
            duration_list.append(this_duration)
            duration_buff+=this_duration
            this_obj.duration = duration.Duration(this_duration)
            symbol_list.append(this_obj)

        if abs(duration_buff-string_duration)>DURATION_EPS_FOR_RECONSTR:
            print('This rhythm string is '+rhythm_string)
            print('string_duration=',string_duration)
            print('duration_buff=',duration_buff)
            assert abs(duration_buff-string_duration)<=DURATION_EPS_FOR_RECONSTR, 'Note duration not compatable with meter!'
    return symbol_list,duration_list


#Discription: the elementwise inverse process of get_melody() for pitches and chords(optional)
#Input type:
#   note_list: [integers](for pitches),
#   duration_list: [integers](corresponding to note_list),
#   chord_alignment_list (optional): [music21 objects](corresponding to note_list),
#   rhythm_string (optional): string(format: rhythm pattern)
#Output type:
#   note_symbol_list: [music21 objects](for pitches),
#   chord_symbol_list (optional if chord_alignment_list): [music21 objects](durations rectified)
#lib: from music21 import *
def note_list_to_symbol_list(note_list,duration_list,chord_alignment_list=None,rhythm_string=None,DURATION_EPS_FOR_RECONSTR = 0.01):
    length=len(note_list)

    note_symbol_list = []
    if chord_alignment_list:
        chord_symbol_list = []
        this_chord = None
        last_chord = None
        this_chord_duration = 0
        chord_cum_duration = 0
    
    this_note = None
    sum_duration = sum(duration_list)

    correctness = True
    if len(note_list)!= len(duration_list):
        correctness = False
    if chord_alignment_list:
        if len(note_list)!= len(chord_alignment_list) or len(duration_list) != len(chord_alignment_list):
            correctness = False
    assert correctness, 'input not same length!'

    if note_list[0] == -3:
        rest_note = note.Rest('')
        rest_note.duration = duration.Duration(0)
        note_symbol_list.append(rest_note)
        if chord_alignment_list:
            chord_symbol_list.append(rest_note)
    elif note_list[0] == -2:
        if rhythm_string:
            if rhythm_string[0] == '|' and abs(duration_list[0])<=DURATION_EPS_FOR_RECONSTR:
                note_symbol_list.append(meter.TimeSignature(rhythm_string[1:]))
                if chord_alignment_list:
                    chord_symbol_list.append(meter.TimeSignature(rhythm_string[1:]))
            else:
                assert False, 'rhythm string or duration not compatable!'
    else:
        if rhythm_string:
            meter_index=rhythm_string.find('|')
            meter_str=rhythm_string[meter_index+1:]

            divide_index=meter_str.find('/')
            element_duration=round(4/float(meter_str[divide_index+1:]),3)
            count_per_measure=int(meter_str[0:divide_index])
            string_duration=count_per_measure*element_duration

            tmp_str = rhythm_string.replace('|'+meter_str,'')
            str_obj_list = tmp_str.split(',')
            duration_buff = 0
            assert abs(sum_duration-string_duration)<=DURATION_EPS_FOR_RECONSTR, 'rhythm string meter not compatable!'
        for i in range(length):
            this_note = note_list[i]
            this_duration = float(duration_list[i])
            if chord_alignment_list:
                this_chord = chord_alignment_list[i]
            if rhythm_string:
                this_type = tmp_str[i][0]
            else:
                this_type = 'N'

            if this_note == -1 or (bool(rhythm_string) and this_type=='H'):
                rest_note = note.Rest('')
                rest_note.duration = duration.Duration(this_duration)
                note_symbol_list.append(rest_note)
            else:
                pitch_note = note.Note(this_note)
                pitch_note.duration = duration.Duration(this_duration)
                note_symbol_list.append(pitch_note)
            if chord_alignment_list:
                if this_chord != last_chord:
                    this_chord_duration=this_duration
                    temp_i = i
                    if temp_i>=length-1:
                        this_chord_duration = sum_duration-chord_cum_duration
                    while temp_i<length-1:
                        if chord_alignment_list[temp_i+1]==this_chord:
                            this_chord_duration += float(duration_list[temp_i+1])
                            temp_i += 1
                        else:
                            break
                    chord_cum_duration+=this_chord_duration
                    this_chord_symbol = this_chord
                    this_chord_symbol.duration = duration.Duration(this_chord_duration)
                    chord_symbol_list.append(this_chord_symbol)
                last_chord = this_chord
    if chord_alignment_list:
        return note_symbol_list,chord_symbol_list
    else:
        return note_symbol_list,None

#Discription: the inverse process of get_melody() for rhythm, with byproduct of duration lists
#Input type:
#   string_list: [strings](format: rhythm pattern),
#   place_for_controller: bool(if True, then generate Rests with duration 0 for controller strings),
#Output type:
#   symbol_stream_tmp: music21 stream(corresponding to string_list),
#   duration_list: [list of floats](corresponding to string_list)
#lib: from music21 import *
#lib: import music21
def translate_rhythm_string_list_into_stream(string_list,place_for_controller=True,DURATION_EPS_FOR_RECONSTR = 0.01):
    symbol_list=[]
    duration_list=[]
    for string in string_list:
        decoded,durations = rhythm_pattern_string_to_duration_list(string,place_for_controller,DURATION_EPS_FOR_RECONSTR)
        if place_for_controller:
            symbol_list = symbol_list+decoded
            duration_list.append(durations)
        elif decoded: #decoded!=[]
            symbol_list = symbol_list+decoded
            duration_list.append(durations)
        else:
            continue

    symbol_stream_tmp=music21.stream.Stream()
    for obj in symbol_list:
        symbol_stream_tmp.append(obj)
    return symbol_stream_tmp,duration_list

#Discription: the inverse process of get_melody() for pitches and chords(optional)
#Input type:
#   note_measure_list: [list of integers](for pitches),
#   note_durations_list: [list of floats](corresponding to note_measure_list),
#   chord_measure_list (optional): [list of music21 objects](corresponding to note_measure_list),
#   rhythm_list (optional): [string](format: rhythm pattern)(corresponding to note_measure_list)
#Output type:
#   note_stream_tmp: music21 stream(for pitches),
#   chord_stream_tmp (optional if chord_measure_list): music21 stream(for chords)
#lib: from music21 import *
def translate_note_list_into_stream(note_measure_list,note_durations_list,chord_measure_list=None,rhythm_list=None,DURATION_EPS_FOR_RECONSTR = 0.01):
    note_list=[]
    if chord_measure_list:
        chord_list=[]

    assert len(note_measure_list) == len(note_durations_list), 'Length not compatable!'
    if chord_measure_list:
        assert len(note_measure_list) == len(chord_measure_list), 'Length not compatable!'
    if rhythm_list:
        assert len(note_measure_list) == len(rhythm_list), 'Length not compatable!'
    
    length = len(note_measure_list)
    for i in range(length):
        note_measure = note_measure_list[i]
        duration_measure = note_durations_list[i]

        if chord_measure_list:
            chord_measure = chord_measure_list[i]
        else:
            chord_measure = None
        if rhythm_list:
            rhythm_str = rhythm_list[i]
        else:
            rhythm_str = None

        note_list_temp,chord_list_temp = \
        note_list_to_symbol_list(note_measure, \
                            duration_measure, \
                            chord_measure, \
                            rhythm_str, DURATION_EPS_FOR_RECONSTR)

        note_list = note_list+note_list_temp
        if chord_measure_list:
            chord_list = chord_list+chord_list_temp

    note_stream_tmp=music21.stream.Stream()
    for obj in note_list:
        note_stream_tmp.append(obj)

    if chord_measure_list:
        chord_stream_tmp=music21.stream.Stream()
        last_obj = chord.Chord()
        for obj in chord_list:
            if obj.id==last_obj.id:
                #dur_sum = obj.duration.quarterLength+last_obj.duration.quarterLength
                #dur_obj = duration.Duration(dur_sum)
                #obj.duration = dur_obj
                #chord_stream_tmp[-1]=obj
                duraion_log = obj.duration
                if type(obj)==chord.Chord:
                    obj = chord.Chord(obj)
                elif type(obj)==note.Note:
                    obj = note.Note(obj)
                elif type(obj)==rest.rest:
                    obj = note.Rest(obj)
                else:
                    duraion_log = duration.Duration(0)
                    obj = note.Rest(obj)
                obj.duration = duraion_log
                chord_stream_tmp.append(obj)
            else:
                chord_stream_tmp.append(obj)
            last_obj = obj
    else:
        chord_stream_tmp=None

    return note_stream_tmp,chord_stream_tmp



