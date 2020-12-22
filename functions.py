#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import datetime



import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, GlobalAveragePooling1D, Dropout,LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
#Building the neural network
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Masking, TimeDistributed, Lambda
from tensorflow.keras.models import Model



from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler



from sklearn.manifold import TSNE
import seaborn as sns
import umap.umap_ as umap



reducer = umap.UMAP()





tsne = TSNE()








def truncate_sequence(sequence,n):
    return [sequence[i:i + n] for i in range(0, len(sequence), n)]




def preprocess_data(list_of_input_paths,output_path,atomic_size,N,counting_method='original_sequence'):
    with open(output_path, "w") as output_handle:
        output_handle.write('sequence' + "\t" + 'class' +'\n')
        if counting_method=='original_sequence':
            class_i=1
            for input_path in list_of_input_paths:
                count=0
                for seq_record in SeqIO.parse(input_path, "fasta"):
                    count+=1  
                    if count<N+1:       
                        sequence = str(seq_record.seq)
                        list_of_atom_seq=truncate_sequence(sequence,atomic_size)
                        for seq in list_of_atom_seq:
                            output_handle.write(seq + "\t" + str(class_i) +'\n')
                        
                class_i+=1
        if counting_method=='truncated_sequence':
            class_i=1
            for input_path in list_of_input_paths:
                count=1
                for seq_record in SeqIO.parse(input_path, "fasta"):
                    sequence = str(seq_record.seq)
                    list_of_atom_seq=truncate_sequence(sequence,atomic_size)
                    for seq in list_of_atom_seq:
                        if count<N+1:       
                            output_handle.write(seq + "\t" + str(class_i) +'\n')
                            count+=1
                class_i+=1
    return 






def binary_preprocess_data(list_of_input_paths,output_path,atomic_size,N,counting_method='original_sequence'):
    '''
    if counting method = original sequence, count number only increases when each one of the truncated sequences
    of an original sequence are added to the dataframe
    
    elif counting method = truncated sequence, count number increases for each one of the truncated sequences
    
    '''
    
    
    
    with open(output_path, "w") as output_handle:
        output_handle.write('sequence' + "\t" + 'class' +'\n')
        if counting_method=='original_sequence':
            class_i=1
            input_path = list_of_input_paths[0]
            count=1
            for seq_record in SeqIO.parse(input_path, "fasta"):
                if count<N+1:       
                    sequence = str(seq_record.seq)
                    list_of_atom_seq=truncate_sequence(sequence,atomic_size)
                    count+=1 
                    for seq in list_of_atom_seq:
                        output_handle.write(seq + "\t" + str(class_i) +'\n')
                        
                        
            class_i+=1
            for input_path in list_of_input_paths[1:]:
                count=0
                for seq_record in SeqIO.parse(input_path, "fasta"):
                    if count<N+1:       
                        sequence = str(seq_record.seq)
                        list_of_atom_seq=truncate_sequence(sequence,atomic_size)
                        count+=1 
                        for seq in list_of_atom_seq:
                            output_handle.write(seq + "\t" + str(class_i) +'\n')
                         
                            

        if counting_method=='truncated_sequence':
            class_i=1
            input_path = list_of_input_paths[0]
            count=1
            for seq_record in SeqIO.parse(input_path, "fasta"):
                if count<N+1:       
                    sequence = str(seq_record.seq)
                    list_of_atom_seq=truncate_sequence(sequence,atomic_size)
                    for seq in list_of_atom_seq:
                        output_handle.write(seq + "\t" + str(class_i) +'\n')
                        count+=1  
                    
            class_i+=1
            for input_path in list_of_input_paths[1:]:
                count=1
                for seq_record in SeqIO.parse(input_path, "fasta"):
                    sequence = str(seq_record.seq)
                    list_of_atom_seq=truncate_sequence(sequence,atomic_size)
                    for seq in list_of_atom_seq:
                        if count<N+1:       
                            output_handle.write(seq + "\t" + str(class_i) +'\n')
                            count+=1
    return 







def one_hot_split(dataframe,atomic_size,test_size=0.25):
    #this function is in charge of one-hot encoding and train test split
    
    sequences=list(dataframe['sequence'])


    #Getting all the base elements
    base=[]
    for seq in sequences:
        for i in list(set(seq)):
            base.append(i)
    base=set(base)
    print(base, len(base))
    
    
    #Add a sequence that contains all the bases to ensure consistent encodings
    for i in range(len(dataframe['sequence'])):
        dataframe['sequence'][i]=''.join(base)+dataframe['sequence'][i]

    sequences=list(dataframe['sequence'])
    labels=list(dataframe['class'])




    #Remove the shorter end pieces 
    L=atomic_size+len(base)
    good_idx=[]
    for i in range(len(sequences)):
        if len(sequences[i])==L:
            good_idx.append(i)

    sequences=[seq for seq in sequences if len(seq)==L]

    labels=[labels[i] for i in good_idx]

    #test if there is no shorter pieces
    for seq in sequences:
        if len(seq)!=atomic_size+len(base):
            print('error in seq length')


    # The LabelEncoder encodes a sequence of bases as a sequence of integers.
    integer_encoder = LabelEncoder()  
    
    # The OneHotEncoder converts an array of integers to a sparse matrix where 
    # each row corresponds to one possible value of each feature.
    one_hot_encoder = OneHotEncoder(categories=[np.arange(len(base))]*1)   
    input_features = []

    for sequence in sequences:
        integer_encoded = integer_encoder.fit_transform(list(sequence))
        integer_encoded = np.array(integer_encoded).reshape(-1, 1)
        one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
        input_features.append(one_hot_encoded.toarray())

    print('Input Shape',np.shape(input_features))

    np.set_printoptions(threshold=40)
    input_features = np.stack(input_features)
    print("Example sequence\n-----------------------")
    print('DNA Sequence #1:\n',sequences[0][:10],'...',sequences[0][-10:])
    print('One hot encoding of Sequence #1:\n',input_features[0].T)

    one_hot_encoder = OneHotEncoder(categories='auto')
    labels = np.array(labels).reshape(-1, 1)
    input_labels = one_hot_encoder.fit_transform(labels).toarray()

    print('Labels:\n',labels.T)
    print('One-hot encoded labels:\n',input_labels.T)


    train_features, test_features, train_labels, test_labels = train_test_split(
        input_features, input_labels, test_size=0.25, random_state=42)


    X_train=train_features
    y_train=train_labels
    X_test=test_features
    y_test=test_labels
    
    print(np.shape(X_train),np.shape(y_train))

    #batch_cutoff_train=len(X_train)%n_batch
    #print(batch_cutoff_train)

    #if batch_cutoff_train>0:

    #    X_train=X_train[:-batch_cutoff_train]
    #    y_train=y_train[:-batch_cutoff_train]

    #    batch_cutoff_test=len(X_test)%n_batch
    #    X_test=X_test[:-batch_cutoff_test]
    #    y_test=y_test[:-batch_cutoff_test]


    #print(np.shape(X_train),np.shape(y_train))
    
    return X_train,y_train,X_test,y_test



def deeper_AE(bottleneck_dim,X_train,lr,activation='tanh'):
    encoder_decoder = Sequential()
    serie_size=X_train.shape[1]
    n_features=X_train.shape[2]
    encoder_decoder.add(LSTM(serie_size, activation=activation, input_shape=(serie_size, n_features), return_sequences=True))
    encoder_decoder.add(LSTM(64, activation=activation, return_sequences=True))
    encoder_decoder.add(LSTM(32, activation=activation, return_sequences=True))
    encoder_decoder.add(LSTM(bottleneck_dim, activation=activation))
    encoder_decoder.add(RepeatVector(serie_size))
    encoder_decoder.add(LSTM(serie_size, activation=activation, return_sequences=True))
    encoder_decoder.add(LSTM(32, activation=activation, return_sequences=True))
    encoder_decoder.add(LSTM(64, activation=activation, return_sequences=True))
    encoder_decoder.add(TimeDistributed(Dense(1)))
    encoder_decoder.summary()
    adam = tf.keras.optimizers.Adam(lr)
    encoder_decoder.compile(loss='mse', optimizer=adam)


    return encoder_decoder


def deeper_AE4(bottleneck_dim,X_train,lr,activation='tanh'):
    encoder_decoder = Sequential()
    serie_size=X_train.shape[1]
    n_features=X_train.shape[2]
    encoder_decoder.add(LSTM(serie_size, activation=activation, input_shape=(serie_size, n_features), return_sequences=True))
    encoder_decoder.add(LSTM(128, activation=activation, return_sequences=True))
    encoder_decoder.add(LSTM(64, activation=activation, return_sequences=True))
    encoder_decoder.add(LSTM(32, activation=activation, return_sequences=True))
    encoder_decoder.add(LSTM(bottleneck_dim, activation=activation))
    encoder_decoder.add(RepeatVector(serie_size))
    encoder_decoder.add(LSTM(serie_size, activation=activation, return_sequences=True))
    encoder_decoder.add(LSTM(32, activation=activation, return_sequences=True))
    encoder_decoder.add(LSTM(64, activation=activation, return_sequences=True))
    encoder_decoder.add(LSTM(128, activation=activation, return_sequences=True))
    encoder_decoder.add(TimeDistributed(Dense(1)))
    encoder_decoder.summary()
    adam = tf.keras.optimizers.Adam(lr)
    encoder_decoder.compile(loss='mse', optimizer=adam)


    return encoder_decoder






def create_model(bottleneck_dim,X_train,lr,activation='tanh'):
    encoder_decoder = Sequential()
    serie_size=X_train.shape[1]
    n_features=X_train.shape[2]
    encoder_decoder.add(LSTM(serie_size, activation=activation, input_shape=(serie_size, n_features), return_sequences=True))
    #encoder_decoder.add(LSTM(64, activation=activation, return_sequences=True))
    encoder_decoder.add(LSTM(32, activation=activation, return_sequences=True))
    encoder_decoder.add(LSTM(bottleneck_dim, activation=activation))
    encoder_decoder.add(RepeatVector(serie_size))
    encoder_decoder.add(LSTM(serie_size, activation=activation, return_sequences=True))
    encoder_decoder.add(LSTM(32, activation=activation, return_sequences=True))
    #encoder_decoder.add(LSTM(64, activation=activation, return_sequences=True))
    encoder_decoder.add(TimeDistributed(Dense(1)))
    encoder_decoder.summary()
    adam = tf.keras.optimizers.Adam(lr)
    encoder_decoder.compile(loss='mse', optimizer=adam)


    return encoder_decoder



def Encoding(list_of_input_paths,output_path,colour,labels,Atomic_size,N,n_class,\
             lr,n_epoch,batch_size,encoder_level,bottleneck_dim,NN_function,binary='False',LOAD='False'):
    
    '''
    bottleneck_dim=dimension of the compressed data
    encoder_level: specify the level of the encoder
    NN_function: The Autoencoder function to be used for data compression
    Atomic_size: A list of Atomic sizes to be used for the experiment. If only wished to use a singular atomic size, 
    input Atomic_size=[atomic_size]
    '''




    Train_encoded=[]  #AutoEncoder compressed/ encoded
    Test_encoded=[]
    
    X_tSNE_encoded_list=[]
    X_tSNE_list=[]
    X_tSNE_test_encoded_list=[]
    X_tSNE_test_list=[]
    
    
    X_train_list=[]
    X_test_list=[]
    Y_train=[]
    Y_test=[]
    
    Umap_encoded=[]  #Umap embedding of the AE encoded data

    '''
    preprocess and store the data for each atomic size respectively, learn 

    '''

    for k in range(len(Atomic_size)):

        if binary=='True':
            binary_preprocess_data(list_of_input_paths,output_path,Atomic_size[k],N,counting_method='truncated_sequence')
            
        else:
            preprocess_data(list_of_input_paths,output_path,Atomic_size[k],N,counting_method='truncated_sequence')





        df = pd.read_table('data/data.txt')
        df['class'].value_counts().sort_index().plot.bar()

        X_train,y_train,X_test,y_test=one_hot_split(df,Atomic_size[k],test_size=0.25)

        X_tSNE=tsne.fit_transform(X_train[:, :, 0])
        X_tSNE_list.append(X_tSNE)
        X_tSNE_test=tsne.fit_transform(X_test[:, :, 0])
        X_tSNE_test_list.append(X_tSNE_test)
        X_train_list.append(X_train)
        X_test_list.append(X_test)


        encoder_decoder = NN_function(bottleneck_dim,X_train,lr,activation='tanh')



        checkpoint_path = "training/cp-actual-{L=%i}-{AE=%s}.ckpt"%(Atomic_size[k],str(NN_function))
        checkpoint_dir = os.path.dirname(checkpoint_path)
        #EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
        EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='loss')
        #latest = tf.train.latest_checkpoint(checkpoint_dir)
        #encoder_decoder.load_weights(latest)
        
        if LOAD=='True':
            encoder_decoder.load_weights(checkpoint_path)
        
        else:
            
            
            encoder_decoder_history=encoder_decoder.fit(X_train, X_train, batch_size,steps_per_epoch= 1, epochs=n_epoch,  callbacks=[EarlyStopping], verbose=0)
            encoder_decoder.save_weights(checkpoint_path)

            print(encoder_decoder_history.history['loss'])
            encoder=Model(inputs=encoder_decoder.inputs, outputs=encoder_decoder.layers[encoder_level].output)



            train_encoded = encoder.predict(X_train)
            Train_encoded.append(train_encoded)
            validation_encoded = encoder.predict(X_test)
            Test_encoded.append(validation_encoded)


            y_train=[np.argmax(i) for i in y_train]
            Y_train.append(y_train)
            y_test=[np.argmax(i) for i in y_test]
            Y_test.append(y_test)


            scaled_encoded = StandardScaler().fit_transform(train_encoded)
            umap_encoded = reducer.fit_transform(scaled_encoded)
            Umap_encoded.append(umap_encoded)



            X_tSNE_encoded=tsne.fit_transform(train_encoded)
            X_tSNE_encoded_list.append(X_tSNE_encoded)
            X_tSNE_test_encoded=tsne.fit_transform(validation_encoded)
            X_tSNE_test_encoded_list.append(X_tSNE_test_encoded)


        
        
    return X_train_list,X_test_list,Y_train,Y_test,Train_encoded,\
        Test_encoded,X_tSNE_encoded_list,X_tSNE_list,X_tSNE_test_encoded_list,\
        X_tSNE_test_list,Umap_encoded
    





