# implementing Apriori algorithm from mlxtend

# conda install -c conda-forge mlxtend
#import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules



Book = pd.read_csv("D:/Training/ExcelR_2/book.csv")


frequent_itemsets = apriori(Book, min_support=0.005, max_len=3,use_colnames = True)


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

frequent_itemsets = apriori(Book, min_support=0.01, max_len=3,use_colnames = True)


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

frequent_itemsets = apriori(Book, min_support=0.03, max_len=4,use_colnames = True)

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