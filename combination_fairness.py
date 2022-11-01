import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import fairness_constraint
import AI
class data_structure():
	def __init__(self,file,protected_attribute_name_list,label_name):
		self.file_name = file
		self.df = pd.DataFrame()
		self.Y_name =  label_name
		self.A_col_name = protected_attribute_name_list
		self.privilage_in_A = np.zeros(len(self.A_col_name))
		self.A_df = pd.DataFrame()
		self.ID =np.zeros(len(self.A_col_name))
	

		data=pd.read_csv(self.file_name)
		data.drop(columns=data.columns[-2:],inplace=True)

		# Making a copy of the data to not make changes in the original file
		df=copy.deepcopy(data)

		self.ID = df["CLIENTNUM"]
		
		df.drop(columns="CLIENTNUM",inplace = True)
		
		self.df = df

		for i in range(len(self.A_col_name)):
			old_name = self.A_col_name[i]
			new_name = "A_"+str(i+1)
			self.df.rename(columns={old_name: new_name}, inplace=True)
		self.data_cleaning_cat_to_num()

		self.A_df = self.df.filter(regex='A_')
		df.drop(df.filter(regex='A_').columns, axis=1, inplace=True)
		self.df.rename(columns={self.Y_name: 'Y'}, inplace=True)
		self.Y = self.df['Y']


	def data_cleaning_cat_to_num(self):
		categorical=self.df.select_dtypes(exclude=['int64','float64']).columns
		for i in categorical:
		    self.df[i]=pd.factorize(self.df[i])[0]

	def data_add_col(self,col_name,value_list):
		#yet to test
		self.df.reset_index(inplace=True, drop=True)
		df = pd.DataFrame({col_name : value_list})
		self.df =pd.concat([self.df,df], axis=1)

	def data_drop_col(self,col_name):
		self.df.drop(columns = [col_name],inplace = True)


	def fairness_per_protected_attr(self,df_conditon):
		if len(df_conditon) != 0:
			df = self.df.query(df_conditon)
		else:
			df = self.df

		count_list = df['R'].value_counts().tolist()
		return min(count_list)/max(count_list) 
		#P(unprivilaged)/P(privilaged) = rel_freq(unpriv)/rel_freq(priv) = freq(unpriv)/freq(priv)
		
	def protected_attribute_reset(self):
		self.df.reset_index(inplace=True, drop=True)
		self.df =pd.concat([self.df,self.A_df], axis=1)
		
	


#might be interesting to see what happen in the intermediate transformation
def disparate_impact(dataset,repair_level):

	dataset.protected_attribute_reset()

	dataset.df = fairness_constraint.disparate_constraint(dataset.df,repair_level,dataset.A_df.columns.tolist(),'Y')
	dataset.df.drop(dataset.df.filter(regex='A_').columns, axis=1, inplace=True)

	return dataset
def independence(dataset,pred,repair_level):
	dataset.data_add_col('R',pred)

	
	dataset.protected_attribute_reset()
	
	dataset.df = fairness_constraint.independence_constraint(dataset.df,repair_level,dataset.A_df.columns.tolist(),'R')
	dataset.df.drop(dataset.df.filter(regex='A_').columns, axis=1, inplace=True)
	dataset.data_drop_col('R')

	return dataset

def seperation(dataset,pred,repair_level):
	dataset.data_add_col('R',pred)
	dataset.protected_attribute_reset()

	dataset.df = fairness_constraint.seperation_constraint(dataset.df,repair_level,dataset.A_df.columns.tolist(),'R')
	dataset.df.drop(dataset.df.filter(regex='A_').columns, axis=1, inplace=True)
	dataset.data_drop_col('R')

	return dataset

def sufficient(dataset,pred,repair_level):
	dataset.data_add_col('R',pred)
	dataset.protected_attribute_reset()

	dataset.df = fairness_constraint.sufficient_constraint(dataset.df,repair_level,dataset.A_df.columns.tolist(),'Y')
	dataset.df.drop(dataset.df.filter(regex='A_').columns, axis=1, inplace=True)
	dataset.data_drop_col('R')

	return dataset


#disparate_impact()
#independence()
#seperation()
#sufficient()

def single_fairness(repair_level):
	file = "BankChurners.csv"
	label_name = "Attrition_Flag"
	protected_attribute_name_list = ["Gender","Education_Level","Marital_Status","Total_Relationship_Count"]
	clean_data = data_structure(file,protected_attribute_name_list,label_name)
	clean_data.data_cleaning_cat_to_num()
	
	pred ,acc= AI.rf(clean_data.df)
	#print(clean_data.df.info())
	pred, acc_after = AI.rf(sufficient(clean_data,pred,repair_level).df)#change here
	#print(clean_data.df.info())
	print("\n")
	return [acc,acc_after]


def combo_12(repair_level):
	file = "BankChurners.csv"
	label_name = "Attrition_Flag"
	protected_attribute_name_list = ["Gender","Education_Level","Marital_Status","Total_Relationship_Count"]
	clean_data = data_structure(file,protected_attribute_name_list,label_name)

	clean_data.data_cleaning_cat_to_num()

	pred ,acc= AI.rf(clean_data.df)
	
	clean_data = disparate_impact(clean_data,repair_level)
	#print(clean_data.df.info())

	
	clean_data = independence(clean_data,pred,repair_level)#change here
	#print(clean_data.df.info())

	pred ,acc_after= AI.rf(clean_data.df)
	return [acc,acc_after]
def combo_42(repair_level):
	file = "BankChurners.csv"
	label_name = "Attrition_Flag"
	protected_attribute_name_list = ["Gender","Education_Level","Marital_Status","Total_Relationship_Count"]
	clean_data = data_structure(file,protected_attribute_name_list,label_name)
	clean_data.data_cleaning_cat_to_num()

	pred ,acc= AI.rf(clean_data.df)

	#print(clean_data.df.info())
	clean_data = sufficient(clean_data,pred,repair_level)
	#print(clean_data.df.info())

################################################################################
	clean_data = independence(clean_data,pred,repair_level)#change here
##################################################################################
	pred ,acc_after= AI.rf(clean_data.df)
	print("\n")
	return [acc,acc_after]

def combo_231(repair_level):
	file = "BankChurners.csv"
	label_name = "Attrition_Flag"
	protected_attribute_name_list = ["Gender","Education_Level","Marital_Status","Total_Relationship_Count"]
	clean_data = data_structure(file,protected_attribute_name_list,label_name)
	
	clean_data.data_cleaning_cat_to_num()

	pred ,acc= AI.rf(clean_data.df)

	
	clean_data = independence(clean_data,pred,repair_level)#change here

	clean_data =seperation(clean_data,pred,repair_level)

	clean_data = disparate_impact(clean_data,repair_level)
	pred ,acc_after= AI.rf(clean_data.df)

	print("\n")
	return [acc,acc_after]

def combo_341(repair_level):
	file = "BankChurners.csv"
	label_name = "Attrition_Flag"
	protected_attribute_name_list = ["Gender","Education_Level","Marital_Status","Total_Relationship_Count"]
	clean_data = data_structure(file,protected_attribute_name_list,label_name)
	clean_data.data_cleaning_cat_to_num()

	pred ,acc= AI.rf(clean_data.df)

	
	#change here
	clean_data=seperation(clean_data,pred,repair_level)

	clean_data =sufficient(clean_data,pred,repair_level)

	clean_data=disparate_impact(clean_data,repair_level)
	pred ,acc_after= AI.rf(clean_data.df)

	print("\n")
	return [acc,acc_after]

#combo_12(1)