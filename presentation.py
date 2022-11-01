import combination_fairness
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import AI

#random generate abunch repair level(uniformally) and plot acc vs fairness
#repeat trial and generate hist plot, hopefully normal like-> CI
def plot_hist(df,name):
	plt.figure()
	sns.histplot(data=df, x="utility", kde=True) 
	plt.title("AI Accuracy" + name)
	plt.ylabel("Count")
	#plt.show()
	plt.savefig(name+".png")

def document(doc_name,acc_list):
	f = open(doc_name, 'w')
	for i in range(len(acc_list)):
		f.write(str(acc_list[i])+' ')
	f.close()


def no_fairness(trials):
	file = "BankChurners.csv"
	label_name = "Attrition_Flag"
	protected_attribute_name_list = ["Gender","Education_Level","Marital_Status","Total_Relationship_Count"]
	clean_data = combination_fairness.data_structure(file,protected_attribute_name_list,label_name)

	clean_data.data_cleaning_cat_to_num()
	
	acc_list = []
	for i in range(trials):
		acc= AI.rf(clean_data.df)[1]
		acc_list.append(acc)

	print('no fairness, mean acc, std acc',np.mean(acc_list),np.std(acc_list))
	document("no_fairness Accuracy trials.txt",acc_list)
	return pd.DataFrame({'utility':acc_list})

#plot_hist(no_fairness(100),"no_fairness")

def disparate(trials,repair_level):
	
	acc_list = []
	for i in range(trials):
		file = "BankChurners.csv"
		label_name = "Attrition_Flag"
		protected_attribute_name_list = ["Gender","Education_Level","Marital_Status","Total_Relationship_Count"]
		clean_data = combination_fairness.data_structure(file,protected_attribute_name_list,label_name)


		acc= AI.rf(combination_fairness.disparate_impact(clean_data,repair_level).df)[1]
		acc_list.append(acc)

	print('disparate impact, mean acc, std acc',np.mean(acc_list),np.std(acc_list))
	document("disparate_impact.txt",acc_list)
	return pd.DataFrame({'utility':acc_list})

#plot_hist(disparate(100,1),"disparate impact removed")

def independence(trials,repair_level):
	
	acc_list = []
	for i in range(trials):
		file = "BankChurners.csv"
		label_name = "Attrition_Flag"
		protected_attribute_name_list = ["Gender","Education_Level","Marital_Status","Total_Relationship_Count"]
		clean_data = combination_fairness.data_structure(file,protected_attribute_name_list,label_name)

		pred = AI.rf(clean_data.df)[0]
		acc= AI.rf(combination_fairness.independence(clean_data,pred,repair_level).df)[1]
		acc_list.append(acc)
	print('independence, mean acc, std acc',np.mean(acc_list),np.std(acc_list))
	document("independence.txt",acc_list)
	return pd.DataFrame({'utility':acc_list})

#plot_hist(independence(100,1),"independence enforced")

def seperation(trials,repair_level):
	
	acc_list = []
	for i in range(trials):
		file = "BankChurners.csv"
		label_name = "Attrition_Flag"
		protected_attribute_name_list = ["Gender","Education_Level","Marital_Status","Total_Relationship_Count"]
		clean_data = combination_fairness.data_structure(file,protected_attribute_name_list,label_name)

		pred = AI.rf(clean_data.df)[0]
		acc= AI.rf(combination_fairness.seperation(clean_data,pred,repair_level).df)[1]
		acc_list.append(acc)
	document("seperation.txt",acc_list)
	print('seperation, mean acc, std acc',np.mean(acc_list),np.std(acc_list))
	return pd.DataFrame({'utility':acc_list})

#plot_hist(seperation(100,1),"seperation enforced")

def sufficient(trials,repair_level):
	
	acc_list = []
	for i in range(trials):
		file = "BankChurners.csv"
		label_name = "Attrition_Flag"
		protected_attribute_name_list = ["Gender","Education_Level","Marital_Status","Total_Relationship_Count"]
		clean_data = combination_fairness.data_structure(file,protected_attribute_name_list,label_name)

		pred = AI.rf(clean_data.df)[0]
		acc= AI.rf(combination_fairness.sufficient(clean_data,pred,repair_level).df)[1]
		acc_list.append(acc)
	document("sufficient.txt",acc_list)
	print('sufficient, mean acc, std acc',np.mean(acc_list),np.std(acc_list))
	return pd.DataFrame({'utility':acc_list})

#plot_hist(sufficient(100,1),"sufficient enforced")

def combo_12(trials,repair_level):
	acc_list = []
	for i in range(trials):
		acc= combination_fairness.combo_12(repair_level)[1]
		acc_list.append(acc)
	document("combo_12.txt",acc_list)
	print('disparate and independence, mean acc, std acc',np.mean(acc_list),np.std(acc_list))
	return pd.DataFrame({'utility':acc_list})

#plot_hist(combo_12(100,1),"disparate impact removed and independence enforced")

def combo_42(trials,repair_level):
	acc_list = []
	for i in range(trials):
		acc= combination_fairness.combo_42(repair_level)[1]
		acc_list.append(acc)
	document("combo_231.txt",acc_list)
	print('independence and sufficient, mean acc, std acc',np.mean(acc_list),np.std(acc_list))
	return pd.DataFrame({'utility':acc_list})

plot_hist(combo_42(100,1),"disparate impact removed and independence enforced")

def combo_231(trials,repair_level):
	acc_list = []
	for i in range(trials):
		acc= combination_fairness.combo_231(repair_level)[1]
		acc_list.append(acc)
	document("combo_231.txt",acc_list)
	print('disparate independence,seperation, mean acc, std acc',np.mean(acc_list),np.std(acc_list))
	return pd.DataFrame({'utility':acc_list})

#plot_hist(combo_231(100,1),"disparate impact removed with independence, seperation enforced")

def combo_341(trials,repair_level):
	acc_list = []
	for i in range(trials):
		acc= combination_fairness.combo_341(repair_level)[1]
		acc_list.append(acc)
	document("combo_341.txt",acc_list)
	print('disparate,seperation,sufficient, mean acc, std acc',np.mean(acc_list),np.std(acc_list))
	return pd.DataFrame({'utility':acc_list})

#plot_hist(combo_341(100,1),"disparate impact removed with independence, seperation enforced")

