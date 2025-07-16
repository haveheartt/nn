// overview
a tiny scalar-valued autograd engine and a neural net library on top of it with PyTorch-like API, done following (Karpathy's tutorial)[https://github.com/karpathy/micrograd]!

backpropagation is an algorithm that lets you efficiently evaluate the gradient of some kind of 
loss function with respect to the weights of a neural network that allows us to iterativaly tune 
the weights to minimize the loss function

fundamentally, it’s a technique for calculating derivatives quickly. And it’s an essential trick to have in your bag, not only in deep learning, but in a wide variety of numerical computing situations.

backpropagation starts at an output of the graph and moves towards the beginning.

it starts at G and its going to go backwards through that expression graph and recursively apply the chain
rule from calculus, so we evaluate the derivative of g with respect to all the internal nodes, but also 
with respect to the inputs. 

and the derivatives are important because they are telling us how the inputs affects the output through 
this mathematical expression.

at each node, it merges all paths which originated at that node, that is, applies the operator ∂g/∂ to every node.

this might feel a bit like dynamic programming. that’s because it is!

nn are just mathematical expressions, they take the input data and the weights and the output are the predictions or the loss function

beyond its use in deep learning, backpropagation is a powerful computational tool in many other areas, 
ranging from weather forecasting to analyzing numerical stability – it just goes by different names.

// derivatives with one input

we dont take the symbolic approach, we dont write the mathematical expression in a piece of paper and apply
the product rule and all the other rules and derive the mathematical expression of the derivative of the 
original function. no one does that, would be a massive expression with ten of thousands of terms.

instead, we are going to look at the definition of derivative, what it measures and what it tells us about
the function

[https://en.wikipedia.org/wiki/Derivative]

its the limit as h goes to zero of (f(a + h) - f(a)) / h

so basically what it is saying is that if you slightly increase a by a small number h, how does the function respond? what is the slope of that point, does it go up or down and by how much?

f(x) = 3*x**2 - 4*x + 5
h = 0000.1
x = 3.0
f(x) = 20
f(x+h), if we slightly nudge x in the positive direction, how does the function respond? in this case will
respond positively! (20.014003...)
f(x + h) - f(x), this is how much the function responded in the positive direction
(f(x + h) - f(x)) / h, we cant forget to normalize by the run (h), so we have the rise over run to get the
slope.

this is of course the numerical approximation of the slope because we have to make h very small to converge
to the exact amount

//derivatives with multiple inputs

h = 0.000001

a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c 
a += h 
d2 = a*b + c 

slope = (d2 - d1) / h, d2 - d1 is how much the function increased when we bumped the specific input that
we are interested in by a tiny amount, and then is normalized by h to get the slope.

lets think about what is happening, d1 will be 4.0, will d2 be a number slightly bigger or lower than 4.0?
and this will tell you the sign of the derivative.

we are bumping a by h, a will be slightly more positive but b is a negative number, so we are going to 'add'
less to d, which in fact happens 

d1=4.0
d2=3.9996...

and this tells you that the slope will be negative, and the exact number of the slope is be -3.0
and you can also convince yourself that -3 is the right answer mathematically and analytically because if
you have a*b + c and you have calculus, differentiating with respect to a gives you just b.

h = 0.000001

a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c 
b += h 
d2 = a*b + c

now by bumping b by a tiny amount, we will be adding more since b is negative, and what is the sensitivity,
what is the slope of that? it should be 2.0, because differentiating d with respect to b give us a!

h = 0.000001

a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c 
c += h 
d2 = a*b + c 

and if c is bumped by a tiny amount, then of course a*b is unnafected and now c becomes slightly a bit higher
and it makes it slightly a bit higher by the exact same amount we added to c, so that tells us that the slope is 1!

// backpropagation -- simple expression

so lets recap some engine.py code, with the Value data structure we are able to build out mathematical
expressions using plus and times, they are scalar valued and thanks to the children property we are able
to do the forward pass and build out a mathematical expression that produces a single output, when we are
able to see the output of the forward pass of an expression we would like to run backpropagation, and in
backpropagation we start at the output and we calculate the gradient until the start of the graph, and 
really what we are computing for every single value is the derivative of every single node, so that we can
know how every single node impacts the output. in a nn setting we'd be very interested in the derivative of
a loss function with respect to the weights of the nn, so that we can know how these weights are 
impacting the loss, so its basically the derivative of the output with respect to some of its leaf nodes and
those leaf nodes will be the weights of the neural net, the other leaf nodes will be the data itself, but
usually we dont use the derivative of the loss function with respect to data because the data is fixed, but
the weights will be iterated on using the gradient information!

def lol():
  
  h = 0.001
  
  a = Value(2.0, label='a')
  b = Value(-3.0, label='b')
  c = Value(10.0, label='c')
  e = a*b; e.label = 'e'
  d = e + c; d.label = 'd'
  f = Value(-2.0, label='f')
  L = d * f; L.label = 'L'
  L1 = L.data
  
  a = Value(2.0, label='a')
  b = Value(-3.0, label='b')
  b.data += h
  c = Value(10.0, label='c')
  e = a*b; e.label = 'e'
  d = e + c; d.label = 'd'
  f = Value(-2.0, label='f')
  L = d * f; L.label = 'L'
  L2 = L.data
  
  // this is an inline gradient check, we are deriving backpropagation and getting the derivative with 
  respect to all the intermediate results, and numerical gradient is just estimating it using small step size
  print((L2 - L1)/h)
  
lol()

adding a small amount h on a, will be measuring the derivative of L with respect to a. And obviously if we
change L by h, that would be effectively 1, that's kind like the base case of derivatives.

now lets continue, we have that L = d * f, and we'd like to know:

dL/dd =? f

if you don't believe me we can derive, because the proof is very straightforward

(f(x+h) - f(x)) / h

((d*h)*f - d*f) / h
(d*f + h*f - d*f) / h 
(h*f) / h 
f

simmetrically, dL/df = d

now the crux of backpropagation, we need to get dL/dc, in other words, the derivative of L with respect to c 
because we already computed the other gradients already.

here is the problem, how do we derive dL/dc? we know the derivative dL/dd, so we know how L is sensitive to
d, but how is L sensitive to c? So if we wiggle c how does that impact L? 

it's very intuitive, if you know the impact that c is having on d, and the impact d is having on L, then you
can somehow figure out how c impacts L.

dd/dc =?    

d = c + e

we can go back to the basics of calculus and derive this

(f(x+h) - f(x)) / h

focusing on c and its effect on d, we can do:

((c+h + e) - (c + e)) / h

(c + h + e - c - e) / h 
h / h = 1

by simmetry also dd/de will be 1.0 also

so basically the derivative of a sum expression is very simple.

this is the local derivative because we have the final output value all at the end of the graph and we are
looking into a little plus node, which doesn't know anything about the rest of the graph, all it knows is that
it did a plus, took c + e and created d, and this plus node also knows the local influence of c on d, in 
other words the derivative of d with respect to c, and it also knows the derivative of d with respect to e,
but that is not what we actually want, that’s just the local derivative, what we actually want is dL/dc.

so we know how d impacts L, and we know how c and e impacts d, how do we put that information together to
write dL/dc?

the answer is of course the chain rule [https://en.wikipedia.org/wiki/Chain_rule]!

"If a variable z depends on the variable y, which itself depends on the variable x (that is, y and z are 
dependent variables), then z depends on x as well, via the intermediate variable y. In this case, the 
chain rule is expressed as dz/dx = dz/dy * dy/dx."

what the chain rule is fundamentally telling you how we chain these derivatives together, so to differentiate
through a function composition we have to apply a multiplication of those derivatives.

"Intuitively, the chain rule states that knowing the instantaneous rate of change of z relative to y and
that of y relative to x allows one to calculate the instantaneous rate of change of z relative to x as the
product of the two rates of change."

"If a car travels twice as fast as a bicycle and the bicycle is four times as fast as a walking man, then 
the car travels 2 × 4 = 8 times as fast as the man."

so this makes it very clear that the correct thing to do is to multiply!

WANT:
dL/dc = (dL/dd) * (dd/dc)

WE KNOW:
dL/dd
dd/dc

because the local gradient of plus nodes are 1.0, we can see that it literally just routes the gradient,
because the plus nodes' local derivatives are just 1.0, and in the chain rule 1.0 * dL/dd is just dL/dd.

so we know that dd/dc and dd/de = -2.0

now that we know dL/de, we can recurse our way to the beggining appliyng the chain rule again

dL/de = -2.0

e = a * b

dL/da = (dL/de) * (de/da)

and that is it, and really all we have done is we iterated through all the nodes one by one and locally applied
the chain rule, we always know the derivative of L with respect to this little output and we look how this
output was produced through some operation and we have the pointers to the children nodes of this operation
and so we know what the local derivatives are and we just multiply them on to the global derivative, so
we just go through and multiply on the local derivatives and that’s what backpropagation is, is just a 
recursive application of the chain rule backards through the computation graph.

// neuron

we want to build neural networks and in the simplest case these are multilayer perceptrons, where we have
layers of neurons which are fully connected to each other. Biologically neurons are very complicated devices
but we have very simple mathematical models of them

here is a simple one:

we have inputs xs and them we have these synapses that have weights on them, so ws are weights. and then
the synapse interacts with the input multiplicatively, so what flows to the cell body is w*x, but we have
multiple inputs, so summ(w*x) + b, this b stands for bias, its the innate trigger of happiness of this neuron,
it can make it a bit more trigger happier or a bit less regardless of the input. we are taking all the w*x
adding the bias and take it through an activation function, its usually a squashing function like a sigmoid
or tanh. and what comes out of the neuron is just the activation function applied to the dot product of the
weights and the inputs.

//backward

_backward is a function that is going to do that little piece of chain rule at each little node that took 
inputs and produced output, here we are going to store how we are going to chain the outputs gradient into
the inputs gradients.

//topological sort

lets think through about what we are actually doing, we laid out a mathematical expression and now we are
trying to go backwards through that expression, this means we never want to call a ._backward for any node
before we have done everything after it, we have to get all its full dependencies. This ordering of graphs
can be achieve using something called topological sort.

A topological sort is an ordering of the nodes of a directed graph such that if there is a path from node A
to node B, then node A appears before node B in the ordering.

is basically a laying out of a graph such that all the edges goes only from left to right. 

we maintain a set of visited nodes and we start at some root node, and starting at it we go through all of
its children and we need to lay them out from left to right. basically starts at x, if its not visited
then it marks as visited and iterates through all its children and calls the same function recursively and
after it went through all its children it adds itself. That's how it guarantee that the root node will only
be in the list when all its children are in the list, that's the invariant that is being maintained

and really what we are doing is just calling _backward on all of the nodes on a topological order.

basically we build a topological graph starting at self take, populating the topological order into the topo
list which is a local variable, then we set self.grad to be 1, and then for each node in the reversed list  
we call _backward

//resume

so what are neural nets? they are these mathematical expressions that take inputs as their data as well as
weights and biases, then we have a mathematical expression for the forward pass followed by a loss function
that tries to measure the accuracy of the predictions, usually they are low when your predictions are matching
your targets, so basically the nn is behaving well, and then we backward the loss, use backpropagation to
get the gradient and then we know how to tune all the parameters to decrease the loss locally, but we have
to iterate this process many times in what is called gradient descent!