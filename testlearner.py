import KNNLearner
import LinRegLearner
import numpy as np
import matplotlib.pyplot as plt

def main():
    #preprocess the data in csv file and return proper data format for further analysis
    def DataProcessor(file):
        data = np.loadtxt(file, delimiter=',')
        split = int(len(data) * 0.6)
        Xtrain = data[:split, :2]
        Ytrain = data[:split, 2]
        Xtest = data[split:, :2]
        Ytest = data[split:, 2]
        return Xtrain, Ytrain, Xtest, Ytest

    def KNNTester(Xtrain, Ytrain, Xtest, k):
        learner = KNNLearner.KNNLearner(k)
        learner.addEvidence(Xtrain, Ytrain)
        return learner.query(Xtest)

    def LinRegTester(Xtrain, Ytrain, Xtest):
        learner = LinRegLearner.LinRegLearner()
        learner.addEvidence(Xtrain, Ytrain)
        return learner.query(Xtest)

    def TestAnalyzer(Ypredict, Ytest):
        RMS = np.sqrt(np.sum(np.square(np.subtract(Ypredict, Ytest))) / len(Ytest))
        corr_coef = np.corrcoef(Ypredict, Ytest)[0][1]
        return RMS, corr_coef

    #test of KNNLearner
    def KNNStatsCalculator(file):
        min_RMS_out_of_sample = 1000000
        min_RMS_corr_coef_out_of_sample = -1
        best_k = -1
        ls_knn_RMS_out_of_sample = []
        ls_knn_RMS_in_sample = []
        ls_k = []
        for k in range(1, 51):
            ls_k.append(k)
            knn_Xtrain, knn_Ytrain, knn_Xtest, knn_Ytest = DataProcessor(file)
            knn_Ypredict = KNNTester(knn_Xtrain, knn_Ytrain, knn_Xtest, k)
            knn_RMS_out_of_sample, knn_corr_coef_out_of_sample = TestAnalyzer(knn_Ypredict, knn_Ytest)
            ls_knn_RMS_out_of_sample.append(knn_RMS_out_of_sample)
            knn_Ypredict_in_sample = KNNTester(knn_Xtrain, knn_Ytrain, knn_Xtrain, k)
            knn_RMS_in_sample, knn_corr_coef_in_sample = TestAnalyzer(knn_Ypredict_in_sample, knn_Ytrain)
            ls_knn_RMS_in_sample.append(knn_RMS_in_sample)
            if knn_RMS_out_of_sample < min_RMS_out_of_sample:
                min_RMS_out_of_sample = knn_RMS_out_of_sample
                min_RMS_corr_coef_out_of_sample = knn_corr_coef_out_of_sample
                best_k = k
        best_k_Ypredict = KNNTester(knn_Xtrain, knn_Ytrain, knn_Xtest, best_k)
        return best_k, min_RMS_out_of_sample, min_RMS_corr_coef_out_of_sample, ls_knn_RMS_out_of_sample, ls_knn_RMS_in_sample, ls_k, best_k_Ypredict

    #test of LinRegLearner
    def LinRegStatsCalculator(file):
        reg_Xtrain, reg_Ytrain, reg_Xtest, reg_Ytest = DataProcessor(file)
        reg_Ypredict = LinRegTester(reg_Xtrain, reg_Ytrain, reg_Xtest)
        reg_RMS, reg_corr_coef = TestAnalyzer(reg_Ypredict, reg_Ytest)
        return reg_RMS, reg_corr_coef, reg_Ypredict

    def ChartCreator(ls_k, in_sample_data_Y, out_of_sample_data_Y, file):
        plt.clf()
        plt.plot(ls_k, in_sample_data_Y)
        plt.plot(ls_k, out_of_sample_data_Y)
        plt.legend(['in-sample'] + ['out-of-sample'])
        plt.xlabel('Value of K')
        plt.ylabel('RMS')
        plt.title('The Relationship between K & RMS Using ' + file)
        plt.savefig(file[:-4] + '.png', format = 'png')

    def ScatterplotCreator(data_predict, data_actual, data_x, plot_name):
        plt.clf()
        plt.scatter(data_x, data_predict, c = 'yellow')
        plt.scatter(data_x, data_actual, c = 'blue')
        plt.xlim((0, 400))
        plt.legend(['predicted value'] + ['actual value'])
        plt.title(plot_name)
        plt.savefig(plot_name + '.png', format = 'png')
    
    file1 = 'data-classification-prob.csv'
    file2 = 'data-ripple-prob.csv'
    file1_best_k, file1_min_RMS_out_of_sample, file1_corr_coef_out_of_sample, file1_ls_knn_RMS_out_of_sample, file1_ls_knn_RMS_in_sample, file1_ls_k, file1_best_k_Ypredict = KNNStatsCalculator(file1)
    
    file2_best_k, file2_min_RMS_out_of_sample, file2_corr_coef_out_of_sample, file2_ls_knn_RMS_out_of_sample, file2_ls_knn_RMS_in_sample, file2_ls_k, file2_best_k_Ypredict = KNNStatsCalculator(file2)
    
    file1_reg_RMS, file1_reg_corr_coef, file1_reg_Ypredict = LinRegStatsCalculator(file1)
    file2_reg_RMS, file2_reg_corr_coef, file2_reg_Ypredict = LinRegStatsCalculator(file2)

    ChartCreator(file1_ls_k, file1_ls_knn_RMS_in_sample, file1_ls_knn_RMS_out_of_sample, file1)
    ChartCreator(file2_ls_k, file2_ls_knn_RMS_in_sample, file2_ls_knn_RMS_out_of_sample, file2)

    file1_Yactual = DataProcessor(file1)[3]
    file2_Yactual = DataProcessor(file2)[3]
    file1_points = [i for i in range(1, len(file1_Yactual) + 1)]
    file2_points = [i for i in range(1, len(file2_Yactual) + 1)]

    knn_plot_name_1 = 'Predicted vs Actual using KNN Learner - dataset1'
    ScatterplotCreator(file1_best_k_Ypredict, file1_Yactual, file1_points, knn_plot_name_1)
    knn_plot_name_2 = 'Predicted vs Actual using KNN Learner - dataset2'
    ScatterplotCreator(file2_best_k_Ypredict, file2_Yactual, file2_points, knn_plot_name_2)

    reg_plot_name_1 = 'Predicted vs Actual using Linear Regression Learner - dataset1'
    ScatterplotCreator(file1_reg_Ypredict, file1_Yactual, file1_points, reg_plot_name_1)
    reg_plot_name_2 = 'Predicted vs Actual using Linear Regression Learner - dataset2'
    ScatterplotCreator(file2_reg_Ypredict, file2_Yactual, file1_points, reg_plot_name_2)

    print 'Test file: ' + file1 + '\n' + 'Result of KNN Learner:\n' + '\tBest k = ' + str(file1_best_k) + '\n\tRMS: ' + str(file1_min_RMS_out_of_sample) + '\n\tCorrelation Coefficient: ' + str(file1_corr_coef_out_of_sample) + '\n' + 'Result of Linear Regression:\n' + '\tRMS: ' + str(file1_reg_RMS) + '\n\tCorrelation Coefficient: ' + str(file1_reg_corr_coef)
    print '=' * 60
    print 'Test file: ' + file2 + '\n' + 'Result of KNN Learner:\n' + '\tBest k = ' + str(file2_best_k) + '\n\tRMS: ' + str(file2_min_RMS_out_of_sample) + '\n\tCorrelation Coefficient: ' + str(file2_corr_coef_out_of_sample) + '\n' + 'Result of Linear Regression:\n' + '\tRMS: ' + str(file2_reg_RMS) + '\n\tCorrelation Coefficient: ' + str(file2_reg_corr_coef)

if __name__ == '__main__':
    main()