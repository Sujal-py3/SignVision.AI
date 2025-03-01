import hand_detector2 as hdm
import cv2
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

#Read and process data
data = pd.read_csv('sign_language.csv')
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

X = data.drop('label', axis=1)
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


def convert_sequences(sequences, label):
    '''
    Converts a list of sequences into a format suitable for a DataFrame.
    Each sequence is a list of frames, and each frame contains the coordinates of hand landmarks.

    Parameters:
    sequences (list): A list of sequences where each sequence represents a 15-frame hand movement.
    label (str): The label to be assigned to each sequence.

    Returns:
    dict: A dictionary containing the converted data, with keys as column names and values as lists of coordinates.
    '''

    #Checks if there are any sequences to add to the dataset
    if sequences:
        data = {}

        #Iterates through each sequence in the list of sequences
        for sequence in sequences:

            #Iterates through each frame in the sequence
            for frame_num in range(len(sequence)):
                #Resets the landmark number at the start of each frame
                landmark_num = 0

                #Iterates through each location coordinate in a frame
                for location_num in range(len(sequence[frame_num])):

                    #Checks if the location coordinates are in the first half of the list (left hand)
                    if location_num in range(0,42):
                        handedness = 'L'

                        #Checks if the location coordinate is odd or even indexed (coordinates alternate x and y's)
                        if location_num % 2 == 0:
                            coordinate = 'x'
                        else:
                            coordinate = 'y'
                    
                    #If it's in the second half, represents the right hand
                    else:  
                        handedness = 'R'

                        #Checks if the location coordinate is odd or even indexed (coordinates alternate x and y's)
                        if location_num % 2 == 0:
                            coordinate = 'x'
                        else:
                            coordinate = 'y'
                    
                    #Sets the column name and adds the value for each column
                    column_name = f'{frame_num}, {handedness}, {landmark_num}, {coordinate}'
                    if column_name in data:
                        data[column_name].append(sequence[frame_num][location_num])
                    else:
                        data[column_name] = [sequence[frame_num][location_num]]

                    #Updates landmark number by location of 
                    if location_num % 2 != 0:
                        landmark_num+= 1
                    if landmark_num > 20:
                        landmark_num = 0

        #Adds the label for each new data entry
        data['label'] = [label] * len(sequences)

        return data

def main():
    '''
    Main function to run the hand gesture recognition system.
    Captures video input, detects hand landmarks, and classifies hand gestures in real-time.

    The function also handles the collection and storage of new gesture data.

    Inputs: None

    Returns: None
    '''

    #Initializes variables, hand detector, and video
    cap = cv2.VideoCapture(0)
    detector = hdm.handDetector()
    sequence_length = 15
    sequences = []
    current_sequence = []
    inactive_frames = 0
    inactivity_reset_length = 3


    #While loop for running the interpreter
    while True:

        #Initialize image from the camera
        success, img = cap.read()
        img = cv2.flip(img, 1)
        key = cv2.waitKey(1) & 0xFF

        #Initializes hand finder and position finder from handDetector class
        img = detector.find_hands(img, draw=False)
        landmarks = detector.find_position(img)
        
        #Checks if the hands are visible on the screen
        if landmarks:

            #Initializes left and right hand location vectors
            left_hand = [0] * 42
            right_hand = [0] * 42

            for hand in landmarks:
                #Sets the handedness and the landmark list for each hand
                handedness = hand[0]
                lmlist = hand[1]

                #Finds the highest and lowest points of each hand to draw the rectangle around the hand
                p1 = (min(lmlist[x][1] for x in range(len(lmlist))) - 25, min(lmlist[x][2] for x in range(len(lmlist))) - 25)
                p2 = (max(lmlist[x][1] for x in range(len(lmlist))) + 25, max(lmlist[x][2] for x in range(len(lmlist))) + 25)
                cv2.rectangle(img, p1, p2, (255, 255, 255), 3)

                #Creates a location vector based on the coordiantes from the landmark list
                location_vector = [coord for lm in lmlist for coord in lm[1:3]]
                if handedness == 'Left':
                    left_hand = location_vector
                elif handedness == 'Right':
                    right_hand = location_vector

            #Combines the two vectors for the left and right hand and adds it to the current sequence
            combined_vector = left_hand + right_hand
            current_sequence.append(combined_vector)

            #Checks if the frame count has reached the desired number
            if len(current_sequence) == sequence_length:

                #Adds the current sequence to the list of sequences
                sequences.append(current_sequence)

                #Reshapes the current sequence and uses it to predict
                sequence_array = np.array(current_sequence).flatten().reshape(1, -1)
                print(model.predict(sequence_array)[0])
                print(model.predict_proba(sequence_array))
                # print('Sequence Copmplete')

                #Resets the current sequence
                current_sequence = []
                

        #If there are no hands detected, update the inactivity timer
        else:
            inactive_frames += 1

            #If inactivity timer reaches desired time, reset the current sequence
            if inactive_frames >= inactivity_reset_length:
                current_sequence = []
                inactive_frames = 0
                        
        #Show the image
        cv2.imshow("Image", img)

        #If q is pressed, stop the program
        if key == ord('q'):
            break
        
        #If c is pressed, update the dataset and stop the program
        if key == ord('c'):
            old_data = pd.read_csv('sign_language.csv')
            new_data = convert_sequences(sequences, 'help')
            new_data = pd.DataFrame(new_data)
            updated_data = pd.concat([old_data, new_data], ignore_index=True)
            updated_data.to_csv('sign_language.csv', index=False)
            with open('recent_data.pkl', 'wb') as f:
                pickle.dump(old_data, f)
            break

#Run the program
# if __name__ == '__main__':
#     main()

#Reset the sign_language data

# with open('hand_gesture_sequences.pkl', 'rb') as f:
    # loaded_sequences = pickle.load(f)

# data = convert_sequences(loaded_sequences, 'hello')
# data = pd.DataFrame(data)
# data.to_csv('sign_language.csv')
