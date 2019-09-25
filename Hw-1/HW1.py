# Craig Campbell
# HW1 
# Machine Learning
# Math 592

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print(tf.__file__)


#showing figure 
x_data = np.load('x_hw.npy')
y_data = np.load('y_hw.npy')
##plt.plot(x_data,y_data)
##plt.show()

#defining variables
a1 = tf.Variable(tf.ones([1]))
a2 = tf.Variable(tf.ones([1]))
f1 = tf.Variable(tf.ones([1]))
f2 = tf.Variable(tf.ones([1]))

#defining model
y = a1 * tf.sin(f1*x_data) + a2 * tf.sin(f2 * x_data)

#define loss function
loss = tf.reduce_mean(tf.square(y-y_data))

#optimizer and trainstep
optimizer= tf.train.AdamOptimizer(0.005)
train_step = optimizer.minimize(loss)


#############
## session ##
#############

#session initiliazation
session = tf.InteractiveSession()
tf.global_variables_initializer().run()

#tensorboard initizalation
tf.summary.scalar('loss',loss)
merged_file = tf.summary.merge_all()
file_writer = tf.summary.FileWriter('777')
file_writer.add_graph(session.graph)

N=10000
for k in range(N):
    session.run(train_step)
    loss_log=session.run(merged_file)
    file_writer.add_summary(loss_log,k)
    if(k%200)==0:
        print("k=",k,"a1",session.run(a1),"f1=",session.run(f1),"a2=",session.run(a2),"f2=",session.run(f2),"loss=",session.run(loss))
A1,F1,A2,F2 = session.run([a1,f1,a2,f2])

# collecting variables 
A1=A1[0]
F1=F1[0]
A2=A2[0]
F2=F2[0]

print("\n The line of this equation is: \n")
print("y(f1,f2)=" + str(A1) + "*sin(" + str(F1) +"X)" + " + " + str(A2) +"*sin(" + str(F2) + "X)")

#model vs reality
y_model = 1.4866351*np.sin(6.0589256*x_data) + 1.4866351*np.sin(6.0589256*x_data)
#plt.plot(x_data,y_data,'r')
#plt.plot(x_data,y_model,'b')
#plt.show()


#value at 0.6*np.pi of model
yvalue = 1.4866351*np.sin(6.0589256*(0.6*(np.pi))) + 1.4866351*np.sin(6.0589256*(0.6*(np.pi)))
print(yvalue)







        
print('Success')
