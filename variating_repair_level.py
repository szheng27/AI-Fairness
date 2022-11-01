import combination_fairness
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import AI

#random generate abunch repair level(uniformally) and plot acc vs fairness
#repeat trial and generate hist plot, hopefully normal like-> CI
def plot_scatter(df,name):
	plt.figure()
	plt.plot(df[0],df[1],'*')
	plt.title("AI Accuracy" + name)
	plt.ylabel("Accuracy")
	plt.xlabel("fairness")
	#plt.show()
	plt.savefig(name+".png")

def document(doc_name,data):
	f = open(doc_name, 'w')
	for i in range(len(data[0])):
		f.write(str(data[0][i])+' '+str(data[1][i])+'\n')
	f.close()


def fairness_return(df,catagory_name,label_name):
    '''
    for (Y,R,A) in binary classification
    where 
        Y the label
        R the prediction(classification)
        A the protected attribute
    for disparity impact:   label_name = Y, catagory_name = A
    for independence:       label_name = R, catagory_name = A

    for seperation:         label_name = R, catagory_name = A where Y=0,1
    for sufficient:         label_name = Y, catagory_name = A where R=0,1
    '''

    return 



def disparate(trials):
	
	acc_list = []
	rp_list = []#want fairness level afterward
	for i in range(trials):
		repair_level = np.random.rand()
		file = "BankChurners.csv"
		label_name = "Attrition_Flag"
		protected_attribute_name_list = ["Gender","Education_Level","Marital_Status","Total_Relationship_Count"]
		clean_data = combination_fairness.data_structure(file,protected_attribute_name_list,label_name)


		acc= AI.rf(combination_fairness.disparate_impact(clean_data,repair_level).df)[1]
		acc_list.append(acc)
		rp_list.append(repair_level)

	return [rp_list,acc_list]

data = disparate(100)
plot_scatter(data,"DI,various repair level")
document("DI,rp_vs_acc",data)


def independence(trials):
	
	acc_list = []
	rp_list = []#want fairness level afterward
	for i in range(trials):
		repair_level = np.random.rand()
		file = "BankChurners.csv"
		label_name = "Attrition_Flag"
		protected_attribute_name_list = ["Gender","Education_Level","Marital_Status","Total_Relationship_Count"]
		clean_data = combination_fairness.data_structure(file,protected_attribute_name_list,label_name)

		pred = AI.rf(clean_data.df)[0]
		acc= AI.rf(combination_fairness.independence(clean_data,pred,repair_level).df)[1]

		acc_list.append(acc)
		rp_list.append(repair_level)

	return [rp_list,acc_list]
data = independence(100)
plot_scatter(data,"ind,various repair level")
document("Ind,rp_vs_acc",data)


def seperation(trials):
	
	acc_list = []
	rp_list = []#want fairness level afterward
	for i in range(trials):
		repair_level = np.random.rand()
		file = "BankChurners.csv"
		label_name = "Attrition_Flag"
		protected_attribute_name_list = ["Gender","Education_Level","Marital_Status","Total_Relationship_Count"]
		clean_data = combination_fairness.data_structure(file,protected_attribute_name_list,label_name)

		pred = AI.rf(clean_data.df)[0]
		acc= AI.rf(combination_fairness.seperation(clean_data,pred,repair_level).df)[1]
		
		acc_list.append(acc)
		rp_list.append(repair_level)

	return [rp_list,acc_list]
data = seperation(100)
plot_scatter(data,"sep,various repair level")
document("sep,rp_vs_acc",data)

def sufficient(trials):
	
	acc_list = []
	rp_list = []#want fairness level afterward
	for i in range(trials):
		repair_level = np.random.rand()
		file = "BankChurners.csv"
		label_name = "Attrition_Flag"
		protected_attribute_name_list = ["Gender","Education_Level","Marital_Status","Total_Relationship_Count"]
		clean_data = combination_fairness.data_structure(file,protected_attribute_name_list,label_name)

		pred = AI.rf(clean_data.df)[0]
		acc= AI.rf(combination_fairness.sufficient(clean_data,pred,repair_level).df)[1]
		
		acc_list.append(acc)
		rp_list.append(repair_level)

	return [rp_list,acc_list]
data = sufficient(100)
plot_scatter(data,"suff,various repair level")
document("suff,rp_vs_acc",data)

#############################################################################################################################
def combo_12(trials):
	acc_list = []
	rp_list = []#want fairness level afterward
	for i in range(trials):
		repair_level = np.random.rand()
		acc= combination_fairness.combo_12(repair_level)[1]
		
		acc_list.append(acc)
		rp_list.append(repair_level)

	return [rp_list,acc_list]

data = combo_12(100)
plot_scatter(data,"1_2,various repair level")
document("1_2,rp_vs_acc",data)

def combo_42(trials):
	acc_list = []
	rp_list = []#want fairness level afterward
	for i in range(trials):
		repair_level = np.random.rand()
		acc= combination_fairness.combo_42(repair_level)[1]
		
		acc_list.append(acc)
		rp_list.append(repair_level)

	return [rp_list,acc_list]

data = combo_42(100)
plot_scatter(data,"4_2,various repair level")
document("4_2,rp_vs_acc",data)

def combo_231(trials):
	acc_list = []
	rp_list = []#want fairness level afterward
	for i in range(trials):
		repair_level = np.random.rand()
		acc= combination_fairness.combo_231(repair_level)[1]
		
		acc_list.append(acc)
		rp_list.append(repair_level)

	return [rp_list,acc_list]

data = combo_231(100)
plot_scatter(data,"2_3_1,various repair level")
document("2_3_1,rp_vs_acc",data)

def combo_341(trials):
	acc_list = []
	rp_list = []#want fairness level afterward
	for i in range(trials):
		repair_level = np.random.rand()
		acc= combination_fairness.combo_341(repair_level)[1]
		
		acc_list.append(acc)
		rp_list.append(repair_level)

	return [rp_list,acc_list]

data = combo_341(100)
plot_scatter(data,"3_4_1,various repair level")
document("3_4_1,rp_vs_acc",data)
