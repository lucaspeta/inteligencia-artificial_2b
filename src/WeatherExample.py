import numpy
import math

from hmmlearn import hmm

modelo1 = hmm.MultinomialHMM(n_components = 3)          #estados -> chuva, sol, nuvens

modelo1.startprob_ = numpy.array([0.5, 0.2, 0.3])      #probabilidades de chuva, sol, nuvens

modelo1.transmat_ = numpy.array([                      #[chuva, sol, nuvens][chuva, sol, nuvens] => matriz de transicoes
    [0.5, 0.3, 0.2],                                #[chuva -> chuva, sol, nuvens]    
    [0.35, 0.45, 0.2],                               #[sol -> chuva, sol, nuvens] 
    [0.2, 0.3, 0.5]                               #[nuvens -> chuva, sol, nuvens] 
    ])

modelo1.emissionprob_ = numpy.array([
    [0.1, 0.4, 0.5],                                #probabilidade de chuva em caminhar, comprar e limpar
    [0.6, 0.3, 0.1],                                #probabilidade de sol em caminhar, comprar e limpar
    [0.2, 0.1, 0.7]                                 #probabilidade de nuvens em caminhar, comprar e limpar
    ])   
math.exp(modelo1.score(numpy.array([[0]]))) #primeira observacao sendo “caminhar” eh a multiplicacao do estado inicia; cp, a matriz de probabilidade 0.2 x 0.1 + 0.5 x 0.6  + 0.3 x 0.5 = 0.30 (30%)
# 0.30000000000000004
math.exp(modelo1.score(numpy.array([[1]])))
# 0.3606
math.exp(modelo1.score(numpy.array([[2]])))
# 0.3400000000000001
math.exp(modelo1.score(numpy.array([[2,2,2]])))
# 0.04590400000000001

logprob, seq = modelo1.decode(numpy.array([[1,2,0]]).transpose()) #{“comprar”, “limpar”, “caminhar”}
print(math.exp(logprob))
print(seq) #{“chuva”, “nuvens”, “sol”}

logprob, seq = modelo1.decode(numpy.array([[2,2,2]]).transpose()) #{“limpar”, “limpar”, “limpar”}
print(math.exp(logprob))
print(seq) #{“nuvens”, “nuvens”, “nuvens”}

from pomegranate import *

modelo2 = HiddenMarkovModel( name="previsao-tempo" )

chuva = State( DiscreteDistribution({ 'caminhar': 0.1, 'comprar': 0.4, 'limpar': 0.5 }), name='Chuva' )
sol = State( DiscreteDistribution({ 'caminhar': 0.6, 'comprar': 0.3, 'limpar': 0.1 }), name='Sol' )
nuvens = State( DiscreteDistribution({ 'caminhar': 0.2, 'comprar': 0.1, 'limpar': 0.7 }), name='Nuvens' )

modelo2.add_transition( modelo2.start, chuva, 0.5 )
modelo2.add_transition( modelo2.start, sol, 0.2)
modelo2.add_transition( modelo2.start, nuvens, 0.3 )

modelo2.add_transition( chuva, chuva, 0.45 )
modelo2.add_transition( chuva, sol, 0.25 )
modelo2.add_transition( chuva, nuvens, 0.15 )
modelo2.add_transition( sol, chuva, 0.3 )
modelo2.add_transition( sol, sol, 0.4 )
modelo2.add_transition( sol, nuvens, 0.15 ) 
modelo2.add_transition( nuvens, chuva, 0.15 )
modelo2.add_transition( nuvens, sol, 0.25 )
modelo2.add_transition( nuvens, nuvens, 0.45 )

modelo2.add_transition( chuva, modelo2.end, 0.15 )
modelo2.add_transition( sol, modelo2.end, 0.15 )
modelo2.add_transition( nuvens, modelo2.end, 0.15 )

modelo2.bake( verbose=True )

sequencia = ['comprar', 'limpar', 'caminhar', 'limpar', 'limpar', 'caminhar', 'limpar' ]

print (math.e**modelo2.forward( sequencia )[ len(sequencia), modelo2.end_index ]) #Now lets check the probability of observing this sequencia.
#1.8545269908e-05

print (math.e**modelo2.forward_backward( sequencia )[1][ 2, modelo2.states.index( chuva ) ]) #Then the probability that Bob will be cleaning a step 3 in this sequencia.
#0.912099070419

print (math.e**modelo2.backward( sequencia )[ 3, modelo2.states.index( sol ) ]) #The probability of the sequencia occurring given it is sol at step 4 in the sequencia.
#0.0004459435

print (" ".join( state.name for i, state in modelo2.maximum_a_posteriori( sequencia )[1] )) #Finally the probable series of states given the above sequencia.
#sol chuva chuva chuva chuva sol chuva