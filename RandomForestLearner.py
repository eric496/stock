import numpy as np
import copy

class RandomForestLearner():
    def __init__(self, k = 3):
        self.k = k
        self.index = 0
        self.tree = None
        self.forest = None
    
    def createTree(self, data):
        #declare the root index of the tree
        root_ix = self.index
        #copy the data so it is passed by value
        data = data.copy()
        #number of rows and columns
        row_count = data.shape[0]
        col_count = data.shape[1]
        #stop and return y value if there is only one row left
        if row_count == 1:
            self.tree[self.index, :] = [-1, data[0, -1], -1, -1]
        else:
            #initialize the left and right child tree and fill in zeros
            lchild = np.zeros((row_count, col_count))
            rchild = np.zeros((row_count, col_count))
            #select a random feature index
            feat_ix = np.random.randint(col_count - 1)
            #if all the values for a particular feature are the same
            if np.array_equal(data[:, feat_ix], [data[0, feat_ix]] * len(data)):
                #convert the node to be a leaf node with the value being the mean of the Ys
                leaf_val = np.mean(data[:, -1])
                self.tree[self.index, :] = [-1, leaf_val, -1, -1]
            else:
                #select two random feature values
                rand_val1 = data[np.random.randint(row_count), feat_ix]
                rand_val2 = data[np.random.randint(row_count), feat_ix]
                #calculate their mean value as split value
                split_val = np.mean([rand_val1, rand_val2])
                #initialize the root node followed by left child
                #set right child as none initialially 
                self.tree[self.index, :] = [feat_ix, split_val, self.index + 1, -1]
                #declare left and right index counter
                left_ix = right_ix = 0
                for i in range(0, len(data)):
                    #store in left child if value is less than split value
                    if data[i, feat_ix] <= split_val:
                        lchild[left_ix, :] = data[i, :]
                        left_ix += 1
                    #store in right child if value is greater than split value
                    else:
                        rchild[right_ix, :] = data[i, :]
                        right_ix += 1
                self.index += 1 
                #build left child tree recursively
                self.createTree(lchild[0:left_ix])
                if(right_ix >= 1):
                    self.index += 1
                    rroot_ix = self.index
                    #build right tree recursively
                    self.createTree(rchild[0:right_ix])
                    #update right child tree index
                    self.tree[root_ix, -1] = rroot_ix

    def addEvidence(self, Xtrain, Ytrain):
        #merge Xtrain and Ytrain into one numpy array
        data = np.zeros((Xtrain.shape[0], Xtrain.shape[1] + 1))
        data[:, :-1] = Xtrain
        data[:, -1] = Ytrain
        #initialize forest and make it empty
        self.forest = []
        #build k random tree
        for i in range (self.k):
            #initialize index
            self.index = 0
            #shuffle data in place
            np.random.shuffle(data)
            #select 60% of data to create the tree
            data_bag = data[: len(data) * 0.6]
            #initialize tree with numpy array structure 
            #four columns: feature index, split value, left child index, right child index
            self.tree = np.zeros((data_bag.size, 4))
            #build tree
            self.createTree(data_bag)
            #add tree to forest
            self.forest.append(self.tree[:self.index + 1])

    def query(self, Xtest):
        #pass data by value
        data = Xtest.copy()
        ls_Yforest = []
        for row in data:
            ls_row_predict = []
            for tree in self.forest:
                row_ix = 0
                feat_ix = 0
                split_val_ix = 1
                left_tree_ix = 2
                right_tree_ix = 3
                #loop until reach leaf node
                while tree[row_ix, feat_ix] != -1:
                    if row[tree[row_ix, 0]] <= tree[row_ix, split_val_ix]:
                        row_ix = tree[row_ix, left_tree_ix]
                    else:
                        row_ix = tree[row_ix, right_tree_ix]
                #add leaf node value to list
                ls_row_predict.append(tree[row_ix, split_val_ix])
            ls_Yforest.append(ls_row_predict)
        ls_Ypredict = []
        #calculate mean value
        for y in ls_Yforest:
            ls_Ypredict.append(np.mean(y))
        return ls_Ypredict