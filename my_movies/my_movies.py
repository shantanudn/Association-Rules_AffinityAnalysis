# implementing Apriori algorithm from mlxtend

  # conda install -c conda-forge mlxtend
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
  
  
# =============================================================================
#    =============================================================================
#  my_movies = pd.read_csv("D:/Training/ExcelR_2/Association_Rules_MarketBasketAnalysis_AffinityAnalysis_RelationshipMining/my_movies/my_movies.csv")
#  
#  my_movies= my_movies.drop(['Sixth Sense', 'Gladiator','LOTR1',], axis=1)
#  =============================================================================
# =============================================================================
  
my_movies = []
  # As the file is in transaction data we will be reading data directly 
  with open("D:/Training/ExcelR_2/Association_Rules_MarketBasketAnalysis_AffinityAnalysis_RelationshipMining/my_movies/my_movies.csv") as f:
      my_movies = f.read()
  
  
  # splitting the data into separate transactions using separator as "\n"
  my_movies = my_movies.split("\n")
  my_movies_list = []
  for i in my_movies:
      my_movies_list.append(i.split(","))
      
 ######################### Removing empty strings
 
 my_movies_list = [[y for y in x if y!='']for x in my_movies_list]
 
 
 ###############
 
 all_my_movies_list = [i for item in my_movies_list for i in item]
 
# print(type(my_movies_list))
#  
#  all_my_movies_list = [i for item in my_movies_list for i in item]
#  
#  my_movies_list = filter(None, my_movies_list)
#  
#  my_movies_list = ''.join(my_movies_list).split() 
#  
#  str_list = filter(None, my_movies_list)
#  
#  my_list = my_movies_list
#  
#  lst = ["He", "is", "so", "", "cool"]
#  result = [x for x in 1st if x]
#
#  array = ["one", "two", "  ", "four", ""]
#  answer = []
#  for item in my_list:
#    if len(item(strip) > 0:
#        answer.append(item)
#    print(answer)
#  
#    
#   mainList = []
###
#  for list in my_list: # loops through the array of lists (lists of lists?)
#    tempList = []
#    for item in my_list: #Loops through each item of the specific loop
#      if item.strip(): # removes whitespace, and checks it is not an empty item
#         tempList.append(item) # add the item to the list of valid options
#         mainList.append(tempList) # Add the list of valid options to the main array/list/answer
#    print(mainList) 
#    
#
###
#  test_list = [[1,2,3,4,5],[2,3,4,5,6,''],[4,5,6,'',''], [6,7,8,9,10,'','','']]
#  
#  for item in test_list:
#      for unit in item:
#          if unit == '':
#              print('this is empty')
#          
#  secondlist = my_movies_list.remove('')
#
#  test_list = [[1,2,3,4,5],[2,3,4,5,6,''],[4,5,6,'',''], [6,7,8,9,10,'','','']]
#  
#ne_list = [[y for y in x if y!='']for x in test_list]
#  
#  
#a1,a2,a3,a4,a5,a6,a7,a8,a9,a10 = my_list
#a=[z for z in x if z!='']
#b=[k for k in x if l!='']
#test_list=a,b


    
  ####################################################
  from collections import Counter
  
  item_frequencies = Counter(all_my_movies_list)
  # after sorting
  #item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])
  item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])
  
  # Storing frequencies and items in separate variables 
  frequencies = list(reversed([i[1] for i in item_frequencies]))
  items = list(reversed([i[0] for i in item_frequencies]))




# barplot of top 10 

import matplotlib.pyplot as plt


    plt.bar(items[0:11], frequencies[0:11])
    plt.xlabel('Movies')
    plt.ylabel('Frequencies')
    plt.xticks(items[0:11], rotation=30)
    plt.title('Most number of times movies watched')
    plt.show()


# Creating Data Frame for the transactions data 

# Purpose of converting all list into Series object Coz to treat each list element as entire element not to separate 
my_movies_series  = pd.DataFrame(pd.Series(my_movies_list))
my_movies_series = my_movies_series.iloc[:10,:] # removing the last empty transaction

my_movies_series.columns = ["transactions"]

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = my_movies_series['transactions'].str.join(sep='*').str.get_dummies(sep='*')

frequent_itemsets = apriori(X, min_support=0.005, max_len=2,use_colnames = True)

# Most Frequent item sets based on support 
# =============================================================================
# frequent_itemsets.sort_values('support',ascending = False,inplace=True)
# plt.bar(left = list(range(1,11)),height = frequent_itemsets.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
# plt.xlabel('item-sets');plt.ylabel('support')
# =============================================================================

    plt.bar(items[0:10], frequent_itemsets.support[0:10])
    plt.xlabel('Items')
    plt.ylabel('Frequencies')
    plt.xticks(items[0:11], rotation=30)
    plt.title('Most number of items purchased')
    plt.show()

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending = False,inplace=True)

 
########################## To eliminate Redudancy in Rules #################################### 
def to_list(i):
    return (sorted(list(i)))


ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)


ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift',ascending=False).head(10)

###Scatter plot

import matplotlib.pyplot as plt
plt.plot(rules_no_redudancy.support,rules_no_redudancy.confidence,"ro");plt.xlabel("support");plt.ylabel("confidence")


### min_support = 0.01 #####################################################################

frequent_itemsets = apriori(X, min_support=0.01, max_len=3,use_colnames = True)

    plt.bar(items[0:10], frequent_itemsets.support[0:40])
    plt.xlabel('Items')
    plt.ylabel('Frequencies')
    plt.xticks(items[0:10], rotation=30)
    plt.title('Most number of items purchased')
    plt.show()

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending = False,inplace=True)

 
########################## To eliminate Redudancy in Rules #################################### 
def to_list(i):
    return (sorted(list(i)))


ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)


ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift',ascending=False).head(10)

# =============================================================================
# ### for max_len = 4 #####################################
# =============================================================================

frequent_itemsets = apriori(X, min_support=0.03, max_len=4,use_colnames = True)

    plt.bar(items[0:11], frequent_itemsets.support[0:11])
    plt.xlabel('Items')
    plt.ylabel('Frequencies')
    plt.xticks(items[0:11], rotation=30)
    plt.title('Most number of items purchased')
    plt.show()

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending = False,inplace=True)

 
########################## To eliminate Redudancy in Rules #################################### 
def to_list(i):
    return (sorted(list(i)))


ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)


ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift',ascending=False).head(10)