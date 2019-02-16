import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class pnn_classifier:

    def __init__(self, filename1, filename2):
        self.filename_train = filename1
        self.filename_test = filename2
        self.test_data = pd.read_csv(self.filename_test)
        self.train_data = pd.read_csv(self.filename_train)
        self.num_class = self.train_data['Label'].unique()
        self.sigma = 1
        self.point_to_classify = []
        self.gaussian_sum_dict = {}
        self.predicted_label = []

      #  print(self.test_data)

    def pnn_algorithm(self):
        group = self.train_data.groupby('Label')



        for i in range(0, len(self.num_class)):
            grouped = group.get_group(i)
            temp_summation = 0.0
            for j in range(0, len(grouped)):
                temp_list = grouped.values[j].tolist()
                temp_sub = np.subtract(self.point_to_classify, temp_list[:-1]).tolist()
                temp_sub = [x**2 for x in temp_sub]
                temp_div = -1 * (sum(temp_sub)/(2*(self.sigma**2)))
                temp_exp = np.exp(temp_div)
                temp_summation = temp_summation + temp_exp
            temp_summation = temp_summation/len(grouped)
            self.gaussian_sum_dict[i] = temp_summation
        self.predicted_label .append(str(max(self.gaussian_sum_dict, key = self.gaussian_sum_dict.get)))
        print(len(self.predicted_label))

    def pnn_test(self):
        for i in range(0, self.test_data.shape[0]):
            self.point_to_classify = []
            temp_test = self.test_data.iloc[[i]].values[0].tolist()
            self.point_to_classify = temp_test[:-1]
            pnn_classifier.pnn_algorithm(self)
        pnn_classifier.performance_analysis(self)

    def performance_analysis(self):
        predicted_label = self.predicted_label
        matrix_dim = len(self.num_class)
        confusion_matrix = np.zeros((matrix_dim, matrix_dim), dtype=int)
        actual_class = self.test_data['Label'].tolist()
        actual_class = [str(x) for x in actual_class]
        test_group = self.test_data.groupby('Label')
        test_group_len = [len(test_group.get_group(x)) for x in range(0,len(self.num_class))]
        print (test_group_len)
        for i in range(0, len(actual_class)):
            if predicted_label[i] == actual_class[i]:
                row = int(predicted_label[i])
                column = int(actual_class[i])
                confusion_matrix[row][column] +=1
            else:
                row = int(predicted_label[i])
                column = int(actual_class[i])
                confusion_matrix[row][column] += 1

        print(confusion_matrix.T)
        total_true_positive = confusion_matrix.T.diagonal().tolist()
        print('Total True Positive = ', total_true_positive)

        total_false_negative = np.subtract(test_group_len, total_true_positive).tolist()
        print('Total False Negative = ', total_false_negative)

        total_false_positive = np.sum(confusion_matrix.T, axis=0).tolist()
        print('Total False Positive = ', total_false_positive)

        total = self.test_data.shape[0]
        total_true_negative = np.add(total_true_positive, total_false_negative).tolist()
        total_true_negative = np.add(total_true_negative, total_true_positive).tolist()
        total_true_negative = [(total - i) for i in total_true_negative]
        print('Total True Negative = ', total_true_negative)

        accuracy = float (sum(total_true_positive)) / self.test_data.shape[0]

        error_rate = 1 - accuracy

        recall = sum(total_true_positive)/(sum(total_true_positive)+ np.mean(total_false_negative))

        specificity = sum(total_true_negative)/ (sum(total_true_negative)+ np.mean(total_false_positive))

        precision = sum(total_true_positive)/ (sum(total_true_positive)+ np.mean(total_false_positive))

        prevalence = (sum(total_true_positive)+ np.mean(total_false_negative))/ total

        false_positive_rate = 1 - specificity

        null_error_rate = (np.mean(total_true_negative)+np.mean(total_false_positive))/total

        f1_score = 2* ((precision*recall)/ (precision + recall))

        roc_recall = np.divide(total_true_positive, np.add(total_true_positive, total_false_negative).tolist()).tolist()
        roc_fpr = np.divide(total_false_positive, np.add(total_true_negative, total_false_positive).tolist()).tolist()

        print ('Accuracy = %.2f'% (accuracy*100)+' %')
        print ('Error Rate = %.2f'% (error_rate*100)+' %')
        print ('Recall = %.2f' % recall)
        print ('False Positive Rate = %.2f' %(false_positive_rate*100)+ '%')
        print('Specificity = %.2f' % specificity)
        print('Precision = %.2f' % precision)
        print('Prevalence = %.2f' % prevalence)
        print('Null Error Rate = %.2f' % (null_error_rate * 100) + '%')
        print('F1 Score = %.2f' % f1_score)

        plt.plot(roc_fpr, roc_recall, color='red', linestyle='-', linewidth=3, marker='o', markerfacecolor='orange', markersize=12 )
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()

    def change_labels(self):
        for i in range(0, len(self.num_class)):
            for j in range(0, len(self.train_data)):
                if self.train_data['Label'][j] == self.num_class[i]:
                    self.train_data['Label'][j] = i\

        for i in range(0, len(self.num_class)):
            for j in range(0, len(self.test_data)):
                if self.test_data['Label'][j] == self.num_class[i]:
                    self.test_data['Label'][j] = i


initialize = pnn_classifier("pendigits_train.csv" , "pendigits_test.csv")
# initialize.change_labels()
initialize.pnn_test()
