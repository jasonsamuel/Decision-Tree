import csv
import math
import os
from scipy.stats import chi2
import numpy

#Creates a data dict from the .CSV file provided
def load_csv_to_header_data(filename):
    #Opening file
    fpath = os.path.join(os.getcwd(), filename)
    f = csv.reader(open(fpath, newline='\n'))

    #Reading all rows in the file
    all_rows = []
    for r in f:
        all_rows.append(r)

    #Manually creating the headers for the attributes
    counter = 0
    headers = []

    while(counter < 60):
        headers.append("Position" + str(counter))
        counter += 1

    headers.append("Base-Pair")
    idx_to_name, name_to_idx = get_header_name_to_idx_maps(headers)
    row = []

    #Splitting the DNA-string into a character array of 60
    for i in all_rows:
        tmp = list(i[0])
        tmp.append(i[1])
        row.append(tmp)

    #Creating the data containing the headers, the train data and the mapping indexes
    data = {
        'header': headers,
        'rows': row,
        'name_to_idx': name_to_idx,
        'idx_to_name': idx_to_name
    }

    return data

#Creates mapping information
def get_header_name_to_idx_maps(headers):
    name_to_idx = {}
    idx_to_name = {}

    for i in range(0, len(headers)):
        name_to_idx[headers[i]] = i
        idx_to_name[i] = headers[i]

    return idx_to_name, name_to_idx

#Gets all the unique attributes values
def get_uniq_values(data):
    #Obtains all the indexes to headers map
    idx_to_name = data['idx_to_name']
    idxs = idx_to_name.keys()
    val_map = {}

    #Creates a set for each mapping
    for idx in iter(idxs):
        val_map[idx_to_name[idx]] = set()

    #Selects unique attributes from all rows
    for data_row in data['rows']:
        for idx in idx_to_name.keys():
            att_name = idx_to_name[idx]
            val = data_row[idx]

            if val not in val_map.keys():
                val_map[att_name].add(val)

    return val_map

#Provides information about the class distribution for a specific attribute
def get_class_labels(data, target):
    #Get both row data containing DNA-strings and the classes associated with the target attribute
    rows = data['rows']
    col_idx = data['name_to_idx'][target]
    labels = {}

    #Calculate how many times a specific class appears in all the rows
    for r in rows:
        val = r[col_idx]

        if val in labels:
            labels[val] = labels[val] + 1

        else:
            labels[val] = 1

    return labels

#Calculate entropy for the classes provided
def entropy(n, labels):
    en = 0

    for label in labels.keys():
        p_x = labels[label] / n
        en += - p_x * math.log(p_x, 2)

    return en

#Create a new partition of the data based on the remaining attributes
def partition_data(data, remaining_attrs):
    partitions = {}
    data_rows = data['rows']
    partition_att_idx = data['name_to_idx'][remaining_attrs]

    #Check in each row for the remaining attributes
    for row in data_rows:
        row_val = row[partition_att_idx]

        #check to see if the currented attribute selected is included in the partition, if it isnt, add it
        if row_val not in partitions.keys():
            partitions[row_val] = {
                'name_to_idx': data['name_to_idx'],
                'idx_to_name': data['idx_to_name'],
                'rows': list()
            }

        #Add data for each remaining attribute
        partitions[row_val]['rows'].append(row)

    return partitions

#Calculate the average entropy for the remaining attrs and as well as partitioning the data based on the target attribute provided
def avg_entropy_w_partitions(data, remaining_attrs, target):
    #Get partitioned data for remaining attributes
    data_rows = data['rows']
    n = len(data_rows)
    partitions = partition_data(data, remaining_attrs)
    avg_ent = 0

    for partition_key in partitions.keys():
        partitioned_data = partitions[partition_key]
        partition_n = len(partitioned_data['rows'])
        partition_labels = get_class_labels(partitioned_data, target)
        partition_entropy = entropy(partition_n, partition_labels)
        avg_ent += partition_n / n * partition_entropy

    return avg_ent, partitions

#Return the class with highest number of appearances
def most_common_label(labels):
    mcl = max(labels, key=lambda k: labels[k])
    return mcl

#Calculate the Chi^2 number for a specific attribute
def chi2Calc(attr, data, target):
    #Variables nAInC = no of attributes in a specific class, nC = no of times class appears for attribute,nA = no of times attributes appears for each each
    chi2Value = 0
    nAInC = []
    nC = 0
    nA = []
    #No of unique attributes is sixbased on train data presented
    nR = 6
    i = 0
    label = ["N", "EI", "IE"]
    labelValues = get_class_labels(data, target)

    #Set values for all three variables for all three classes
    while i < 3:
        tmp = label[i]
        if tmp not in labelValues.keys():
            lValue = 1
        else:
            lValue = labelValues[tmp]

        nAInC.append(lValue)
        nC += lValue
        nA.append(lValue * nR)
        i += 1

    i = 0

    #Calculate Chi^2 value using the standard formula
    while i < 3:
        chi2Value += ((nAInC[i] - ((nC * nA[i]) / (nC + nA[i]))) ** 2) / ((nC * nA[i]) / (nC + nA[i]))
        i += 1

    #Multipy by nR since we cant check for each attribute
    chi2Value *= nR

    return chi2Value

#ID3 algorithm that creates a decision tree
def id3(data, uniqs, attrs, target):
    labels = get_class_labels(data, target)
    node = {}

    #Check to see if there is only one outcome/classification
    if len(labels.keys()) == 1:
        node['label'] = next(iter(labels.keys()))
        return node

    #Check to see if we are done
    if len(attrs) == 0:
        node['label'] = most_common_label(labels)
        return node

    n = len(data['rows'])
    ent = entropy(n, labels)
    max_info_gain = None
    max_info_gain_att = None
    max_info_gain_partitions = None

    #Selecting Attribute with Max Information Gain
    for a in attrs:
        avg_ent, partitions = avg_entropy_w_partitions(data, a, target)
        info_gain = ent - avg_ent
        if max_info_gain is None or info_gain > max_info_gain:
            max_info_gain = info_gain
            max_info_gain_att = a
            max_info_gain_partitions = partitions

    #No information gained i.e The remianing attributes all have no information gain
    if max_info_gain is None:
        node['label'] = most_common_label(labels)
        return node

    #Modify current list of remaining attributes by removing the attribute with the maximum information gain
    node['attribute'] = max_info_gain_att
    node['nodes'] = {}
    remaining_attrs = list(attrs)
    remaining_attrs.remove(max_info_gain_att)
    uniq_attrs = uniqs[max_info_gain_att]

    #Generate a decision tree for each attribute if they pass the Chi^2 test
    for attr in uniq_attrs:
        attrChi2 = chi2Calc(attr, data, target)
        attrChi2 = int( attrChi2 * (10**20))

        #degree of freedom  (no of attributes - 1) * (no of classes - 1) classes will always be 3 and attributes 4
        df = 10
        attrChi2P = chi2.ppf(0.00, df)
        attrChi2P = int( attrChi2P * (10**20))

        if attrChi2 >= attrChi2P:
            if attr not in max_info_gain_partitions.keys():
                node['nodes'][attr] = {'label': most_common_label(labels)}
                continue

            partition = max_info_gain_partitions[attr]
            node['nodes'][attr] = id3(partition, uniqs, remaining_attrs, target)

    return node

#Print decision tree (Could either the enite tee or the amount decisions to be taken, currently set to print just the number od decisions)
def print_tree(root):
    stack = []
    rules = list()

    #Checking each node for the leaf direction
    def traverse(node, stack, rules):
        if 'label' in node:
            stack.append(' THEN ' + node['label'])
            rules.append(''.join(stack))
            stack.pop()

        elif 'attribute' in node:
            ifnd = 'IF ' if not stack else ' AND '
            stack.append(ifnd + node['attribute'] + ' EQUALS ')

            for subnode_key in node['nodes']:
                stack.append(subnode_key)
                traverse(node['nodes'][subnode_key], stack, rules)
                stack.pop()

            stack.pop()

    traverse(root, stack, rules)
    #print(os.linesep.join(rules))
    print(str(len(rules)) + " possible decisions could be made to determine if the class is IE, EI and N")

#Oppotional function that removes illegal characters i.e characters that are not "A", "C", "T" and "G" from DNA-string
def preProc(csvfile, mode):
    acceptable = ["A", "C", "T", "G"]
    index = 0
    rows = []

    #Opening .CSV file to be read
    fpath = os.path.join(os.getcwd(), csvfile)
    fs = csv.reader(open(fpath, newline='\n'))

    #Check each row for illegal character, if found, replace it with one of the legal characters
    for r in fs:
        counter = 0
        proc = ""
        tmp = list(r[0])

        while counter < 60:
            if tmp[counter] == "D":
                tmp[counter] == "T"
            elif tmp[counter] not in acceptable:
                tmp[counter] = acceptable[index % 4]
                index += 1

            counter += 1

        for e in tmp:
            proc += e
        tmp.clear()
        tmp.append(proc)
        if mode == "train":
            tmp.append(r[1])
        rows.append(tmp)

    #New modified file
    newFile = "modified" + csvfile

    #Writing new modified file
    with open(newFile, mode = "w", newline = "") as test:
        test_writer = csv.writer(test, delimiter = ",", quoting=csv.QUOTE_MINIMAL)
        counter = 0
        for r in rows:
            test_writer.writerow(r)

#Classifies the test .CSV file provided with a decision tree (root)
def classify(csvfile, root, attrs):
    #Opening test file
    fpath = os.path.join(os.getcwd(), csvfile)
    fs = csv.reader(open(fpath, newline='\n'))

    all_rows = []
    rows = []
    headerMap = {}
    counter = 0

    #creating mappings for headers
    for i in attrs:
        headerMap[i] = counter
        counter += 1

    #Reading each DNA-string into two lists. One reads the string, the other reads the string as a character array
    for r in fs:
        tmp1 = list(r[0])
        all_rows.append(tmp1)
        rows.append(r)

    results = []

    #Using the decision tree, trace the decision path for each DNA-string
    for r in all_rows:
        tmp = headerMap[root['attribute']]
        val = r[tmp]
        attr = root['nodes'][val]

        #if current attribute has children i.e we havent reached the final decision, advance to the children and check if it has the final decision
        if 'label' not in attr:
            results.append(check(r, attr, headerMap))

        else:
            results.append(root['nodes'][val]['label'])

    #Write the result of the decision tree into the second column of the test .CSV file
    with open("testClassifiedWithChiSplitting.csv", mode = "w", newline = "") as test:
        test_writer = csv.writer(test, delimiter = ",", quoting=csv.QUOTE_MINIMAL)
        test_writer.writerow(["id","class"])
        counter = 0

        for r in rows:
            tmp2 = [str(counter + 2001)]
            tmp2.append(results[counter])
            counter += 1
            test_writer.writerow(tmp2)

    print(results)

#Check if current attr in row has a final decision in the root
def check(row, root, headerMap):
    result = ""

    #GEt current attribute to be checked
    tmp = headerMap[root['attribute']]
    val = row[tmp]

    #Check if there no final decision at current node and it doesnt habve any child
    if val not in root['nodes']:
        if val == "T" and "D" in root['nodes']:
            val = "D"

        else:
            return "N"

    attr = root['nodes'][val]

    #if current attribute has children i.e we havent reached the final decision, advance to the children and check if it has the final decision
    if 'label' not in attr:
        result = check(row, attr, headerMap)
    else:
        result = root['nodes'][val]['label']

    return result

#preProc("train.csv", "train")
#preProc("testing.csv", "test")
dat = load_csv_to_header_data("train.csv")
attrs = list(dat['header'])
attrs.remove("Base-Pair")
root = id3(dat, get_uniq_values(dat), attrs, "Base-Pair")
print(get_uniq_values(dat))

print_tree(root)

classify("testing.csv", root, attrs)