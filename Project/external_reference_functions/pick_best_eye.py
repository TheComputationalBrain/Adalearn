
from preprocess_eyedata import pupil
import os


def pick_best_eye(filename, list_subject_selected, from_session, to_session):

    """ pick best eye :.
            - filename: name of file in the directory (supersedes the one in initialization)
            - list_subject_selected: list of the participants from who you want to know if data are better
              with the left or right eye
            - nb_session: the number of session per participant
        It returns:
            - A list of string {'left','right'} for each participant from the list_subject_selected.
            It will simply sum the amount of NaN per eye (left or right) for each session of a same participant in order
            to return the best eye (the eye which contains the less amount of NaN data throught all the sessions).
            
        IMPORTANT : This function is only related to the amount of NaN in the diameter columns. It selects the best eye
        for studying pupil diameter and not position (x,y)
        """

    # initialize the class
    data_tob = pupil()
    list_best_eye = []

    for subject in list_subject_selected:
        compteur_nan_left = 0
        compteur_nan_right = 0
        for session in range(from_session, to_session+1):

            # import the recording in python
            data_tob.import_data(filename,events_warner='MSG', eye = 'right', col=[0, 1, 2], col_name=['x_pos', 'y_pos', 'diameter'], eyetracker='tobii')

            # preprocess the data (see help for further options)
            data_tob.preprocess(plot= False)

            compteur_nan_left += data_tob.nb_diam_nan_left
            compteur_nan_right += data_tob.nb_diam_nan_right

        if compteur_nan_left > compteur_nan_right:
            list_best_eye.append('right')
        else:
            list_best_eye.append('left')
            
    print(" list_best_eye = ", list_best_eye)
    return list_best_eye

