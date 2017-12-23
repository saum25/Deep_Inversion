'''
Created on 22 Dec 2017

@author: Saumitra
Detailed analysis of the results
'''

import numpy as np

def result_analysis(pred_before, pred_after, area_per_instance, gen_error, mt, class_threshold, error_threshold, debug_flag):
    
    # quantify the classification performance after masking
    ground = (np.asarray(pred_before))>class_threshold
    pred = (np.asarray(pred_after))>class_threshold
    class_change = np.zeros(len(ground), dtype = 'bool')
    
    count_pass = 0
    count_fail = 0
    count_cc_lt = 0
    count_cnc_lt = 0
    count_cc_gt = 0
    count_cnc_gt = 0
    
    # abs change in prediction
    abs_pred_error = np.abs(np.asarray(pred_before) - np.asarray(pred_after)) # 1d float array
    
    # recall calculation
    for i in range(len(ground)):
        if ground[i]==pred[i]: # recall
            count_pass +=1
            #class_change[i] = False not needed as its a bool array of false know
        else:
            count_fail +=1
            class_change [i]= True

    for i in range(len(ground)):
        if (gen_error[i]<= error_threshold):
            if class_change[i]==True:
                count_cc_lt +=1
            else:
                count_cnc_lt +=1
        elif (gen_error[i]> error_threshold):
            if class_change[i]==True:
                count_cc_gt +=1
            else:
                count_cnc_gt +=1
    if debug_flag:
        print("Total instances:%d" %(count_pass+ count_fail))
        print("Number of fails:%d" %(count_fail))
        print("Average area: %f" %(sum(area_per_instance)/len(area_per_instance)))
        print("Distribution of instances: cc_lt[%d] cnc_lt[%d] cc_gt[%d] cnc_gt[%d]" %(count_cc_lt, count_cnc_lt, count_cc_gt, count_cnc_gt))
    
    # combine class change and prediction change information in one 2-d array    
    analysis_array = np.vstack((abs_pred_error, class_change))
        
    # save the final results in each iteration (govern by threshold) as a tuple
    return analysis_array, (mt, count_pass+count_fail, count_fail, round(sum(area_per_instance)/len(area_per_instance), 2), count_cc_lt, count_cnc_lt, count_cc_gt, count_cnc_gt)




