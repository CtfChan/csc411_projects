def nn_model_two_weight_batch(X, Y, w1, w2, w1_val, w2_val, num_iterations=10, learning_rate=1, print_cost=False):
    n = X.shape[1]
    n_h = 10 #hidden layer neurons
    
    #initialize weight and bias matrix
    W0 = np.random.randn(n, n_h) * 0.01
    b0 = np.zeros(shape=(1, n_h))

    #W0 = W0f
    #b0 = b0f
    
    #initialize w1 and w2
    W0[w1[0], w1[1]] = w1_val
    W0[w2[0], w2[1]] = w2_val
    
    cost_list = []
    W0_list = [] #now stores tuple of w values
    b0_list = []
    
    #first value
    W0_list.append((w1_val, w2_val))
    
    mini_batch_size = 32
    X_batches, Y_batches = generate_mini_batches(X, Y, mini_batch_size=64)

    for i in range(0, num_iterations):
        
        for j in range(len(X_batches)):
            X_batch = X_batches[j]
            Y_batch = Y_batches[j]
            
            #Forward Propagation
            out = forward_prop(X_batch,W0,b0)  

            #Backward Propogation
            dW0 = compute_grad(out, X_batch, Y_batch)
            #db0 = compute_grad_bias(out, X, Y)

            #Update dW0 and db0
            dW0_new = np.zeros_like(W0)
            db0_new = np.zeros_like(b0)
            dW0_new[w1[0], w1[1]] = dW0[w1[0], w1[1]]
            dW0_new[w2[0], w2[1]] = dW0[w2[0], w2[1]]

            #Update 
            W0 = W0 - (learning_rate * dW0_new)
            b0 = b0 - (learning_rate * db0_new)

        #Compute Cost (print every iteration)
        if (print_cost == True) and (i % 10 == 0):
            print("Cost after iteration {}: {}".format(i, compute_cost_new(out,Y_batch)))

            cost_list.append(compute_cost_new(out,Y_batch))
            w1_val, w2_val =  dW0_new[w1[0], w1[1]], dW0_new[w2[0], w2[1]]
            W0_list.append((w1_val, w2_val))

            print("W1_val: ", w1_val)
            print("W2_val: ", w2_val)

        elif (print_cost == False) and (i % 10 == 0):
            cost_list.append(compute_cost_new(out,Y_batch))
            w1_val, w2_val =  dW0_new[w1[0], w1[1]], dW0_new[w2[0], w2[1]]
            W0_list.append((w1_val, w2_val))
        
    return cost_list, W0_list, b0_list





def nn_model_two_weight(X, Y, w1, w2, w1_val, w2_val, 
                        W0, b0, num_iterations=10, 
                        learning_rate=1, print_cost=False,):
    n = X.shape[1]
    n_h = 10 #hidden layer neurons
    
    W0 = np.copy(W0)
    b0 = np.copy(b0)

    #initialize w1 and w2
    W0[w1[0], w1[1]] = w1_val
    W0[w2[0], w2[1]] = w2_val
    
    cost_list = []
    W0_list = [] #now stores tuple of w values
    b0_list = []
    
    #first value
    W0_list.append((w1_val, w2_val))
    
    for i in range(0, num_iterations):
        
        #Forward Propagation
        out = forward_prop(X,W0,b0)  

        #Backward Propogation
        dW0 = compute_grad(out, X, Y)
        #db0 = compute_grad_bias(out, X, Y)

        #Update dW0 and db0
        dW0_new = np.zeros_like(W0)
        db0_new = np.zeros_like(b0)
        dW0_new[w1[0], w1[1]] = dW0[w1[0], w1[1]]
        dW0_new[w2[0], w2[1]] = dW0[w2[0], w2[1]]

        #print("dw1: ", dW0[w1[0], w1[1]])
        #print("dw2: ", dW0[w2[0], w1[1]])
        
        #Update 
        W0 = W0 - (learning_rate * dW0_new)
        b0 = b0 - (learning_rate * db0_new)

        #Compute Cost (print every iteration)
        if (print_cost == True) and (i % 1 == 0):
            print("Cost after iteration {}: {}".format(i, compute_cost_new(out,Y)))

            cost_list.append(compute_cost_new(out,Y))
            w1_val, w2_val =  dW0_new[w1[0], w1[1]], dW0_new[w2[0], w2[1]]
            W0_list.append((w1_val, w2_val))

            print("W1_val: ", w1_val)
            print("W2_val: ", w2_val)
            print("Accuracy on training set: ", compute_accuracy(X, Y, W0, b0))
            

        elif (print_cost == False) and (i % 1 == 0):
            cost_list.append(compute_cost_new(out,Y))
            w1_val, w2_val =  dW0_new[w1[0], w1[1]], dW0_new[w2[0], w2[1]]
            W0_list.append((w1_val, w2_val))

    return cost_list, W0_list, b0_list




def nn_model_two_weight_mom(X, Y, w1, w2, w1_val, w2_val, W0, b0,
                            num_iterations=10, learning_rate=1, print_cost=False):
    n = X.shape[1]
    n_h = 10 #hidden layer neurons
    
    #initialize weight and bias matrix
    #W0 = np.random.randn(n, n_h) * 0.01
    #b0 = np.zeros(shape=(1, n_h))
    
    W0 = np.copy(W0)
    b0 = np.copy(b0)
    
    #initialize w1 and w2
    W0[w1[0], w1[1]] = w1_val
    W0[w2[0], w2[1]] = w2_val
    
    #initialize parameters for momentum GD
    vdW0 = np.zeros_like(W0)
    vdb0 = np.zeros_like(b0)
    beta = 0.99
    
    cost_list = []
    W0_list = [] #now stores tuple of w values
    b0_list = []
    
    #first value
    W0_list.append((w1_val, w2_val))
    
    for i in range(0, num_iterations):
        
        #Forward Propagation
        out = forward_prop(X,W0,b0)  

        #Backward Propogation
        dW0 = compute_grad(out, X, Y)
        #db0 = compute_grad_bias(out, X, Y)

        #Update dW0 and db0
        dW0_new = np.zeros_like(W0)
        db0_new = np.zeros_like(b0)
        dW0_new[w1[0], w1[1]] = dW0[w1[0], w1[1]]
        dW0_new[w2[0], w2[1]] = dW0[w2[0], w2[1]]
        
        vdW0 = (beta*vdW0) + (1-beta)*dW0_new
        #vdb0 = (beta*vdb0) + (1-beta)*db0_new
        
        #Update 
        W0 = W0 - (learning_rate * vdW0)
        #b0 = b0 - (learning_rate * vdb0)

   
        #Compute Cost (print every iteration)
        if (print_cost == True) and (i % 1 == 0):
            print("Cost after iteration {}: {}".format(i, compute_cost_new(out,Y)))

            cost_list.append(compute_cost_new(out,Y))
            w1_val, w2_val =  dW0_new[w1[0], w1[1]], dW0_new[w2[0], w2[1]]
            W0_list.append((w1_val, w2_val))

            print("W1_val: ", w1_val)
            print("W2_val: ", w2_val)
            print("====" * 5)

        elif (print_cost == False) and (i % 1 == 0):
            cost_list.append(compute_cost_new(out,Y))
            w1_val, w2_val =  dW0_new[w1[0], w1[1]], dW0_new[w2[0], w2[1]]
            W0_list.append((w1_val, w2_val))

    return cost_list, W0_list, b0_list


W0f = np.copy(W0_GD2[-1])
b0f = np.copy(b0_GD2[-1])
print(argmax(W0f[:,7]))
print(W0f[argmax(W0f[:,7]),7])


print(max(W_list))
print(min(W_list))
print(average(W_list))
print(len(W_list))

plt.plot([a for a, b in W_list[1:]], [b for a,b in W_list[1:]], 'bo')
#plt.plot([a for a, b in W_list_mom[1:]], [b for a,b in W_list_mom[1:]], 'r+')

plt.show()


#np.random.seed(2)
# x1 = np.random.randint(7, 17)
# y1 = np.random.randint(7, 17)
# int1 = (y1*28) + x1

# x2 = np.random.randint(7, 17)
# y2 = np.random.randint(7, 17)
# int2 = (y2*28) + x2

# x = int1
# y = int2
# print("int1, int2: ")
# print(int1, int2)
#print("x ", x)
#print("y ", y)




_, W_list, _ = nn_model_two_weight(X_train, Y_train, w1, w2, w1_val, w2_val, W0f, b0f, 
                                   num_iterations=10, learning_rate=3, print_cost=True)

print("===" * 5)
