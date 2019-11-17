from pomegranate import *
import random
import math

random.seed(0)

model = HiddenMarkovModel( name="Tennis-Palying" )

sunny= State( DiscreteDistribution({'D01':0.9,'D02':0.9,'D03':0.1,'D04':0.1,'D05':0.1,'D06':0.1,'D07':0.1,'D08':0.9,'D09':0.9,'D10':0.1,'D11':0.9,'D12':0.1,'D13':0.1,'D14':0.1 }), name='sunny')
overcast= State( DiscreteDistribution({'D01':0.1,'D02':0.1,'D03':0.9,'D04':0.1,'D05':0.1,'D06':0.1,'D07':0.9,'D08':0.1,'D09':0.1,'D10':0.1,'D11':0.1,'D12':0.9,'D13':0.9,'D14':0.9 }), name='overcast')
rain= State( DiscreteDistribution({'D01':0.1,'D02':0.1,'D03':0.1,'D04':0.9,'D05':0.9,'D06':0.9,'D07':0.1,'D08':0.1,'D09':0.1,'D10':0.9,'D11':0.1,'D12':0.1,'D13':0.1,'D14':0.9 }), name='rain')
hot= State( DiscreteDistribution({'D01':0.9,'D02':0.9,'D03':0.9,'D04':0.1,'D05':0.1,'D06':0.1,'D07':0.1,'D08':0.1,'D09':0.1,'D10':0.1,'D11':0.1,'D12':0.1,'D13':0.9,'D14':0.1 }), name='hot')
mild= State( DiscreteDistribution({'D01':0.1,'D02':0.1,'D03':0.1,'D04':0.9,'D05':0.1,'D06':0.1,'D07':0.1,'D08':0.9,'D09':0.1,'D10':0.9,'D11':0.9,'D12':0.9,'D13':0.1,'D14':0.9 }), name='mild')
cool= State( DiscreteDistribution({'D01':0.1,'D02':0.1,'D03':0.1,'D04':0.1,'D05':0.9,'D06':0.9,'D07':0.9,'D08':0.1,'D09':0.9,'D10':0.1,'D11':0.1,'D12':0.1,'D13':0.1,'D14':0.1 }), name='cool')
high= State( DiscreteDistribution({'D01':0.9,'D02':0.9,'D03':0.9,'D04':0.9,'D05':0.1,'D06':0.1,'D07':0.1,'D08':0.9,'D09':0.1,'D10':0.1,'D11':0.1,'D12':0.9,'D13':0.1,'D14':0.9 }), name='high')
normal= State( DiscreteDistribution({'D01':0.1,'D02':0.1,'D03':0.1,'D04':0.1,'D05':0.9,'D06':0.9,'D07':0.9,'D08':0.1,'D09':0.9,'D10':0.9,'D11':0.9,'D12':0.1,'D13':0.9,'D14':0.1 }), name='normal')
weak= State( DiscreteDistribution({'D01':0.9,'D02':0.1,'D03':0.9,'D04':0.9,'D05':0.9,'D06':0.1,'D07':0.1,'D08':0.9,'D09':0.9,'D10':0.9,'D11':0.1,'D12':0.1,'D13':0.9,'D14':0.1 }), name='weak')
strong= State( DiscreteDistribution({'D01':0.1,'D02':0.9,'D03':0.1,'D04':0.1,'D05':0.1,'D06':0.9,'D07':0.9,'D08':0.1,'D09':0.1,'D10':0.1,'D11':0.9,'D12':0.9,'D13':0.1,'D14':0.9 }), name='strong')
yes= State( DiscreteDistribution({'D01':0.1,'D02':0.1,'D03':0.9,'D04':0.9,'D05':0.9,'D06':0.1,'D07':0.9,'D08':0.1,'D09':0.9,'D10':0.9,'D11':0.9,'D12':0.9,'D13':0.9,'D14':0.1 }), name='yes')
no= State( DiscreteDistribution({'D01':0.9,'D02':0.9,'D03':0.1,'D04':0.1,'D05':0.1,'D06':0.9,'D07':0.1,'D08':0.9,'D09':0.1,'D10':0.1,'D11':0.1,'D12':0.1,'D13':0.1,'D14':0.9 }), name='no')

model.add_transition( model.start, yes, 0.64 )
model.add_transition( model.start, no, 0.36 )

model.add_transition( yes, yes, 0.33 )
model.add_transition( yes, no, 0.6 )
model.add_transition( no, yes, 0.33 )
model.add_transition( no, no, 0.6 )


#model.add_transition( strong, yes, 0.33 )
#model.add_transition( strong, no, 0.6 )
#model.add_transition( high, yes, 0.33 )
#model.add_transition( high, no, 0.8 )
#model.add_transition( sunny, yes, 0.22 )
#model.add_transition( sunny, no, 0.6 )
#model.add_transition( cool, yes, 0.33 )
#model.add_transition( cool, no, 0.2 )

model.add_transition( yes, model.end, 0.1 )
model.add_transition( no, model.end, 0.1 )

model.bake( verbose=True )

sequence = [ 'D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09', 'D10', 'D11', 'D12', 'D13', 'D14' ]

print (math.e**model.forward( sequence )[ len(sequence), model.end_index ])

print (math.e**model.forward_backward( sequence )[1][ 2, model.states.index( no ) ])

print (" ".join( state.name for i, state in model.maximum_a_posteriori( sequence )[1] ))
