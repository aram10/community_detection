??

??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??	
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

: *
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

: *
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
: *
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
history
encoder
decoder
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
y
	layer_with_weights-0
	layer-0

	variables
trainable_variables
regularization_losses
	keras_api
y
layer_with_weights-0
layer-0
	variables
trainable_variables
regularization_losses
	keras_api

0
1
2
3

0
1
2
3
 
?
	variables
non_trainable_variables
trainable_variables
metrics
layer_metrics
regularization_losses

layers
layer_regularization_losses
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api

0
1

0
1
 
?

	variables
 non_trainable_variables
trainable_variables
!metrics
"layer_metrics
regularization_losses

#layers
$layer_regularization_losses
h

kernel
bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api

0
1

0
1
 
?
	variables
)non_trainable_variables
trainable_variables
*metrics
+layer_metrics
regularization_losses

,layers
-layer_regularization_losses
JH
VARIABLE_VALUEdense_4/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_4/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_5/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_5/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1
 

0
1

0
1
 
?
	variables
.non_trainable_variables
trainable_variables
/metrics
0layer_metrics
regularization_losses

1layers
2layer_regularization_losses
 
 
 

	0
 

0
1

0
1
 
?
%	variables
3non_trainable_variables
&trainable_variables
4metrics
5layer_metrics
'regularization_losses

6layers
7layer_regularization_losses
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:????????? *
dtype0*
shape:????????? 
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_2763736
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_2764246
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_2764268??
?
?
.__inference_sequential_4_layer_call_fn_2763994

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_27632812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_2764151K
9dense_4_kernel_regularizer_square_readvariableop_resource: 
identity??0dense_4/kernel/Regularizer/Square/ReadVariableOp?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_4_kernel_regularizer_square_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_4/kernel/Regularizer/Square?
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Const?
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum?
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_4/kernel/Regularizer/mul/x?
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul?
IdentityIdentity"dense_4/kernel/Regularizer/mul:z:01^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp
?
?
I__inference_sequential_5_layer_call_and_return_conditional_losses_2763450

inputs!
dense_5_2763438: 
dense_5_2763440: 
identity??dense_5/StatefulPartitionedCall?0dense_5/kernel/Regularizer/Square/ReadVariableOp?
dense_5/StatefulPartitionedCallStatefulPartitionedCallinputsdense_5_2763438dense_5_2763440*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_27634372!
dense_5/StatefulPartitionedCall?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_2763438*
_output_shapes

: *
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_5/kernel/Regularizer/Square?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/Const?
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_5/StatefulPartitionedCall1^dense_5/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
P
0__inference_dense_4_activity_regularizer_2763235

activation
identityL
SigmoidSigmoid
activation*
T0*
_output_shapes
:2	
Sigmoidr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
Mean/reduction_indicese
MeanMeanSigmoid:y:0Mean/reduction_indices:output:0*
T0*
_output_shapes
:2
Mean[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2
	Maximum/yc
MaximumMaximumMean:output:0Maximum/y:output:0*
T0*
_output_shapes
:2	
Maximum[
	truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
	truediv/xa
truedivRealDivtruediv/x:output:0Maximum:z:0*
T0*
_output_shapes
:2	
truedivA
LogLogtruediv:z:0*
T0*
_output_shapes
:2
LogS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
mul/xM
mulMulmul/x:output:0Log:y:0*
T0*
_output_shapes
:2
mulS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xQ
subSubsub/x:output:0Maximum:z:0*
T0*
_output_shapes
:2
sub_
truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2
truediv_1/xc
	truediv_1RealDivtruediv_1/x:output:0sub:z:0*
T0*
_output_shapes
:2
	truediv_1G
Log_1Logtruediv_1:z:0*
T0*
_output_shapes
:2
Log_1W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2	
mul_1/xU
mul_1Mulmul_1/x:output:0	Log_1:y:0*
T0*
_output_shapes
:2
mul_1J
addAddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:2
add>
RankRankadd:z:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????2
rangeK
SumSumadd:z:0range:output:0*
T0*
_output_shapes
: 2
SumW
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
mul_2/xV
mul_2Mulmul_2/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mul_2L
IdentityIdentity	mul_2:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::D @

_output_shapes
:
$
_user_specified_name
activation
?$
?
J__inference_autoencoder_2_layer_call_and_return_conditional_losses_2763709
input_1&
sequential_4_2763684: "
sequential_4_2763686:&
sequential_5_2763690: "
sequential_5_2763692: 
identity

identity_1??0dense_4/kernel/Regularizer/Square/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?$sequential_4/StatefulPartitionedCall?$sequential_5/StatefulPartitionedCall?
$sequential_4/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_4_2763684sequential_4_2763686*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_27633472&
$sequential_4/StatefulPartitionedCall?
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall-sequential_4/StatefulPartitionedCall:output:0sequential_5_2763690sequential_5_2763692*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_27634932&
$sequential_5/StatefulPartitionedCall?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_4_2763684*
_output_shapes

: *
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_4/kernel/Regularizer/Square?
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Const?
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum?
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_4/kernel/Regularizer/mul/x?
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_5_2763690*
_output_shapes

: *
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_5/kernel/Regularizer/Square?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/Const?
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
IdentityIdentity-sequential_5/StatefulPartitionedCall:output:01^dense_4/kernel/Regularizer/Square/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity-sequential_4/StatefulPartitionedCall:output:11^dense_4/kernel/Regularizer/Square/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : : : 2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall:P L
'
_output_shapes
:????????? 
!
_user_specified_name	input_1
?$
?
J__inference_autoencoder_2_layer_call_and_return_conditional_losses_2763681
input_1&
sequential_4_2763656: "
sequential_4_2763658:&
sequential_5_2763662: "
sequential_5_2763664: 
identity

identity_1??0dense_4/kernel/Regularizer/Square/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?$sequential_4/StatefulPartitionedCall?$sequential_5/StatefulPartitionedCall?
$sequential_4/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_4_2763656sequential_4_2763658*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_27632812&
$sequential_4/StatefulPartitionedCall?
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall-sequential_4/StatefulPartitionedCall:output:0sequential_5_2763662sequential_5_2763664*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_27634502&
$sequential_5/StatefulPartitionedCall?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_4_2763656*
_output_shapes

: *
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_4/kernel/Regularizer/Square?
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Const?
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum?
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_4/kernel/Regularizer/mul/x?
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_5_2763662*
_output_shapes

: *
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_5/kernel/Regularizer/Square?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/Const?
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
IdentityIdentity-sequential_5/StatefulPartitionedCall:output:01^dense_4/kernel/Regularizer/Square/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity-sequential_4/StatefulPartitionedCall:output:11^dense_4/kernel/Regularizer/Square/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : : : 2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall:P L
'
_output_shapes
:????????? 
!
_user_specified_name	input_1
?A
?
I__inference_sequential_4_layer_call_and_return_conditional_losses_2763937

inputs8
&dense_4_matmul_readvariableop_resource: 5
'dense_4_biasadd_readvariableop_resource:
identity

identity_1??dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulinputs%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAddy
dense_4/SigmoidSigmoiddense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_4/Sigmoid?
#dense_4/ActivityRegularizer/SigmoidSigmoiddense_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2%
#dense_4/ActivityRegularizer/Sigmoid?
2dense_4/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 24
2dense_4/ActivityRegularizer/Mean/reduction_indices?
 dense_4/ActivityRegularizer/MeanMean'dense_4/ActivityRegularizer/Sigmoid:y:0;dense_4/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
:2"
 dense_4/ActivityRegularizer/Mean?
%dense_4/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2'
%dense_4/ActivityRegularizer/Maximum/y?
#dense_4/ActivityRegularizer/MaximumMaximum)dense_4/ActivityRegularizer/Mean:output:0.dense_4/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
:2%
#dense_4/ActivityRegularizer/Maximum?
%dense_4/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2'
%dense_4/ActivityRegularizer/truediv/x?
#dense_4/ActivityRegularizer/truedivRealDiv.dense_4/ActivityRegularizer/truediv/x:output:0'dense_4/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
:2%
#dense_4/ActivityRegularizer/truediv?
dense_4/ActivityRegularizer/LogLog'dense_4/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
:2!
dense_4/ActivityRegularizer/Log?
!dense_4/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_4/ActivityRegularizer/mul/x?
dense_4/ActivityRegularizer/mulMul*dense_4/ActivityRegularizer/mul/x:output:0#dense_4/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
:2!
dense_4/ActivityRegularizer/mul?
!dense_4/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!dense_4/ActivityRegularizer/sub/x?
dense_4/ActivityRegularizer/subSub*dense_4/ActivityRegularizer/sub/x:output:0'dense_4/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
:2!
dense_4/ActivityRegularizer/sub?
'dense_4/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2)
'dense_4/ActivityRegularizer/truediv_1/x?
%dense_4/ActivityRegularizer/truediv_1RealDiv0dense_4/ActivityRegularizer/truediv_1/x:output:0#dense_4/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
:2'
%dense_4/ActivityRegularizer/truediv_1?
!dense_4/ActivityRegularizer/Log_1Log)dense_4/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
:2#
!dense_4/ActivityRegularizer/Log_1?
#dense_4/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2%
#dense_4/ActivityRegularizer/mul_1/x?
!dense_4/ActivityRegularizer/mul_1Mul,dense_4/ActivityRegularizer/mul_1/x:output:0%dense_4/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
:2#
!dense_4/ActivityRegularizer/mul_1?
dense_4/ActivityRegularizer/addAddV2#dense_4/ActivityRegularizer/mul:z:0%dense_4/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
:2!
dense_4/ActivityRegularizer/add?
!dense_4/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_4/ActivityRegularizer/Const?
dense_4/ActivityRegularizer/SumSum#dense_4/ActivityRegularizer/add:z:0*dense_4/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_4/ActivityRegularizer/Sum?
#dense_4/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#dense_4/ActivityRegularizer/mul_2/x?
!dense_4/ActivityRegularizer/mul_2Mul,dense_4/ActivityRegularizer/mul_2/x:output:0(dense_4/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_4/ActivityRegularizer/mul_2?
!dense_4/ActivityRegularizer/ShapeShapedense_4/Sigmoid:y:0*
T0*
_output_shapes
:2#
!dense_4/ActivityRegularizer/Shape?
/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_4/ActivityRegularizer/strided_slice/stack?
1dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_4/ActivityRegularizer/strided_slice/stack_1?
1dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_4/ActivityRegularizer/strided_slice/stack_2?
)dense_4/ActivityRegularizer/strided_sliceStridedSlice*dense_4/ActivityRegularizer/Shape:output:08dense_4/ActivityRegularizer/strided_slice/stack:output:0:dense_4/ActivityRegularizer/strided_slice/stack_1:output:0:dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_4/ActivityRegularizer/strided_slice?
 dense_4/ActivityRegularizer/CastCast2dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_4/ActivityRegularizer/Cast?
%dense_4/ActivityRegularizer/truediv_2RealDiv%dense_4/ActivityRegularizer/mul_2:z:0$dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_4/ActivityRegularizer/truediv_2?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_4/kernel/Regularizer/Square?
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Const?
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum?
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_4/kernel/Regularizer/mul/x?
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul?
IdentityIdentitydense_4/Sigmoid:y:0^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity)dense_4/ActivityRegularizer/truediv_2:z:0^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
/__inference_autoencoder_2_layer_call_fn_2763653
input_1
unknown: 
	unknown_0:
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_autoencoder_2_layer_call_and_return_conditional_losses_27636272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:????????? 
!
_user_specified_name	input_1
?"
?
I__inference_sequential_4_layer_call_and_return_conditional_losses_2763413
input_3!
dense_4_2763392: 
dense_4_2763394:
identity

identity_1??dense_4/StatefulPartitionedCall?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
dense_4/StatefulPartitionedCallStatefulPartitionedCallinput_3dense_4_2763392dense_4_2763394*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_27632592!
dense_4/StatefulPartitionedCall?
+dense_4/ActivityRegularizer/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *9
f4R2
0__inference_dense_4_activity_regularizer_27632352-
+dense_4/ActivityRegularizer/PartitionedCall?
!dense_4/ActivityRegularizer/ShapeShape(dense_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_4/ActivityRegularizer/Shape?
/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_4/ActivityRegularizer/strided_slice/stack?
1dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_4/ActivityRegularizer/strided_slice/stack_1?
1dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_4/ActivityRegularizer/strided_slice/stack_2?
)dense_4/ActivityRegularizer/strided_sliceStridedSlice*dense_4/ActivityRegularizer/Shape:output:08dense_4/ActivityRegularizer/strided_slice/stack:output:0:dense_4/ActivityRegularizer/strided_slice/stack_1:output:0:dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_4/ActivityRegularizer/strided_slice?
 dense_4/ActivityRegularizer/CastCast2dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_4/ActivityRegularizer/Cast?
#dense_4/ActivityRegularizer/truedivRealDiv4dense_4/ActivityRegularizer/PartitionedCall:output:0$dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_4/ActivityRegularizer/truediv?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_2763392*
_output_shapes

: *
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_4/kernel/Regularizer/Square?
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Const?
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum?
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_4/kernel/Regularizer/mul/x?
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul?
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity'dense_4/ActivityRegularizer/truediv:z:0 ^dense_4/StatefulPartitionedCall1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:????????? 
!
_user_specified_name	input_3
?
?
.__inference_sequential_4_layer_call_fn_2763289
input_3
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_27632812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:????????? 
!
_user_specified_name	input_3
?
?
/__inference_autoencoder_2_layer_call_fn_2763870
x
unknown: 
	unknown_0:
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_autoencoder_2_layer_call_and_return_conditional_losses_27635712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:????????? 

_user_specified_nameX
?$
?
J__inference_autoencoder_2_layer_call_and_return_conditional_losses_2763627
x&
sequential_4_2763602: "
sequential_4_2763604:&
sequential_5_2763608: "
sequential_5_2763610: 
identity

identity_1??0dense_4/kernel/Regularizer/Square/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?$sequential_4/StatefulPartitionedCall?$sequential_5/StatefulPartitionedCall?
$sequential_4/StatefulPartitionedCallStatefulPartitionedCallxsequential_4_2763602sequential_4_2763604*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_27633472&
$sequential_4/StatefulPartitionedCall?
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall-sequential_4/StatefulPartitionedCall:output:0sequential_5_2763608sequential_5_2763610*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_27634932&
$sequential_5/StatefulPartitionedCall?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_4_2763602*
_output_shapes

: *
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_4/kernel/Regularizer/Square?
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Const?
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum?
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_4/kernel/Regularizer/mul/x?
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_5_2763608*
_output_shapes

: *
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_5/kernel/Regularizer/Square?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/Const?
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
IdentityIdentity-sequential_5/StatefulPartitionedCall:output:01^dense_4/kernel/Regularizer/Square/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity-sequential_4/StatefulPartitionedCall:output:11^dense_4/kernel/Regularizer/Square/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : : : 2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall:J F
'
_output_shapes
:????????? 

_user_specified_nameX
?
?
I__inference_sequential_5_layer_call_and_return_conditional_losses_2764044

inputs8
&dense_5_matmul_readvariableop_resource: 5
'dense_5_biasadd_readvariableop_resource: 
identity??dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_5/BiasAddy
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_5/Sigmoid?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_5/kernel/Regularizer/Square?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/Const?
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
IdentityIdentitydense_5/Sigmoid:y:0^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?d
?
J__inference_autoencoder_2_layer_call_and_return_conditional_losses_2763796
xE
3sequential_4_dense_4_matmul_readvariableop_resource: B
4sequential_4_dense_4_biasadd_readvariableop_resource:E
3sequential_5_dense_5_matmul_readvariableop_resource: B
4sequential_5_dense_5_biasadd_readvariableop_resource: 
identity

identity_1??0dense_4/kernel/Regularizer/Square/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?+sequential_4/dense_4/BiasAdd/ReadVariableOp?*sequential_4/dense_4/MatMul/ReadVariableOp?+sequential_5/dense_5/BiasAdd/ReadVariableOp?*sequential_5/dense_5/MatMul/ReadVariableOp?
*sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype02,
*sequential_4/dense_4/MatMul/ReadVariableOp?
sequential_4/dense_4/MatMulMatMulx2sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_4/dense_4/MatMul?
+sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_4/dense_4/BiasAdd/ReadVariableOp?
sequential_4/dense_4/BiasAddBiasAdd%sequential_4/dense_4/MatMul:product:03sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_4/dense_4/BiasAdd?
sequential_4/dense_4/SigmoidSigmoid%sequential_4/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_4/dense_4/Sigmoid?
0sequential_4/dense_4/ActivityRegularizer/SigmoidSigmoid sequential_4/dense_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????22
0sequential_4/dense_4/ActivityRegularizer/Sigmoid?
?sequential_4/dense_4/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2A
?sequential_4/dense_4/ActivityRegularizer/Mean/reduction_indices?
-sequential_4/dense_4/ActivityRegularizer/MeanMean4sequential_4/dense_4/ActivityRegularizer/Sigmoid:y:0Hsequential_4/dense_4/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
:2/
-sequential_4/dense_4/ActivityRegularizer/Mean?
2sequential_4/dense_4/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.24
2sequential_4/dense_4/ActivityRegularizer/Maximum/y?
0sequential_4/dense_4/ActivityRegularizer/MaximumMaximum6sequential_4/dense_4/ActivityRegularizer/Mean:output:0;sequential_4/dense_4/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
:22
0sequential_4/dense_4/ActivityRegularizer/Maximum?
2sequential_4/dense_4/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<24
2sequential_4/dense_4/ActivityRegularizer/truediv/x?
0sequential_4/dense_4/ActivityRegularizer/truedivRealDiv;sequential_4/dense_4/ActivityRegularizer/truediv/x:output:04sequential_4/dense_4/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
:22
0sequential_4/dense_4/ActivityRegularizer/truediv?
,sequential_4/dense_4/ActivityRegularizer/LogLog4sequential_4/dense_4/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
:2.
,sequential_4/dense_4/ActivityRegularizer/Log?
.sequential_4/dense_4/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.sequential_4/dense_4/ActivityRegularizer/mul/x?
,sequential_4/dense_4/ActivityRegularizer/mulMul7sequential_4/dense_4/ActivityRegularizer/mul/x:output:00sequential_4/dense_4/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
:2.
,sequential_4/dense_4/ActivityRegularizer/mul?
.sequential_4/dense_4/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??20
.sequential_4/dense_4/ActivityRegularizer/sub/x?
,sequential_4/dense_4/ActivityRegularizer/subSub7sequential_4/dense_4/ActivityRegularizer/sub/x:output:04sequential_4/dense_4/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
:2.
,sequential_4/dense_4/ActivityRegularizer/sub?
4sequential_4/dense_4/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?26
4sequential_4/dense_4/ActivityRegularizer/truediv_1/x?
2sequential_4/dense_4/ActivityRegularizer/truediv_1RealDiv=sequential_4/dense_4/ActivityRegularizer/truediv_1/x:output:00sequential_4/dense_4/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
:24
2sequential_4/dense_4/ActivityRegularizer/truediv_1?
.sequential_4/dense_4/ActivityRegularizer/Log_1Log6sequential_4/dense_4/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
:20
.sequential_4/dense_4/ActivityRegularizer/Log_1?
0sequential_4/dense_4/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?22
0sequential_4/dense_4/ActivityRegularizer/mul_1/x?
.sequential_4/dense_4/ActivityRegularizer/mul_1Mul9sequential_4/dense_4/ActivityRegularizer/mul_1/x:output:02sequential_4/dense_4/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
:20
.sequential_4/dense_4/ActivityRegularizer/mul_1?
,sequential_4/dense_4/ActivityRegularizer/addAddV20sequential_4/dense_4/ActivityRegularizer/mul:z:02sequential_4/dense_4/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
:2.
,sequential_4/dense_4/ActivityRegularizer/add?
.sequential_4/dense_4/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_4/dense_4/ActivityRegularizer/Const?
,sequential_4/dense_4/ActivityRegularizer/SumSum0sequential_4/dense_4/ActivityRegularizer/add:z:07sequential_4/dense_4/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2.
,sequential_4/dense_4/ActivityRegularizer/Sum?
0sequential_4/dense_4/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??22
0sequential_4/dense_4/ActivityRegularizer/mul_2/x?
.sequential_4/dense_4/ActivityRegularizer/mul_2Mul9sequential_4/dense_4/ActivityRegularizer/mul_2/x:output:05sequential_4/dense_4/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 20
.sequential_4/dense_4/ActivityRegularizer/mul_2?
.sequential_4/dense_4/ActivityRegularizer/ShapeShape sequential_4/dense_4/Sigmoid:y:0*
T0*
_output_shapes
:20
.sequential_4/dense_4/ActivityRegularizer/Shape?
<sequential_4/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential_4/dense_4/ActivityRegularizer/strided_slice/stack?
>sequential_4/dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_4/dense_4/ActivityRegularizer/strided_slice/stack_1?
>sequential_4/dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_4/dense_4/ActivityRegularizer/strided_slice/stack_2?
6sequential_4/dense_4/ActivityRegularizer/strided_sliceStridedSlice7sequential_4/dense_4/ActivityRegularizer/Shape:output:0Esequential_4/dense_4/ActivityRegularizer/strided_slice/stack:output:0Gsequential_4/dense_4/ActivityRegularizer/strided_slice/stack_1:output:0Gsequential_4/dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6sequential_4/dense_4/ActivityRegularizer/strided_slice?
-sequential_4/dense_4/ActivityRegularizer/CastCast?sequential_4/dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-sequential_4/dense_4/ActivityRegularizer/Cast?
2sequential_4/dense_4/ActivityRegularizer/truediv_2RealDiv2sequential_4/dense_4/ActivityRegularizer/mul_2:z:01sequential_4/dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 24
2sequential_4/dense_4/ActivityRegularizer/truediv_2?
*sequential_5/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_5_dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02,
*sequential_5/dense_5/MatMul/ReadVariableOp?
sequential_5/dense_5/MatMulMatMul sequential_4/dense_4/Sigmoid:y:02sequential_5/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_5/dense_5/MatMul?
+sequential_5/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_5/dense_5/BiasAdd/ReadVariableOp?
sequential_5/dense_5/BiasAddBiasAdd%sequential_5/dense_5/MatMul:product:03sequential_5/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_5/dense_5/BiasAdd?
sequential_5/dense_5/SigmoidSigmoid%sequential_5/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
sequential_5/dense_5/Sigmoid?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_4/kernel/Regularizer/Square?
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Const?
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum?
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_4/kernel/Regularizer/mul/x?
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_5_dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_5/kernel/Regularizer/Square?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/Const?
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
IdentityIdentity sequential_5/dense_5/Sigmoid:y:01^dense_4/kernel/Regularizer/Square/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp,^sequential_4/dense_4/BiasAdd/ReadVariableOp+^sequential_4/dense_4/MatMul/ReadVariableOp,^sequential_5/dense_5/BiasAdd/ReadVariableOp+^sequential_5/dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity6sequential_4/dense_4/ActivityRegularizer/truediv_2:z:01^dense_4/kernel/Regularizer/Square/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp,^sequential_4/dense_4/BiasAdd/ReadVariableOp+^sequential_4/dense_4/MatMul/ReadVariableOp,^sequential_5/dense_5/BiasAdd/ReadVariableOp+^sequential_5/dense_5/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : : : 2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp2Z
+sequential_4/dense_4/BiasAdd/ReadVariableOp+sequential_4/dense_4/BiasAdd/ReadVariableOp2X
*sequential_4/dense_4/MatMul/ReadVariableOp*sequential_4/dense_4/MatMul/ReadVariableOp2Z
+sequential_5/dense_5/BiasAdd/ReadVariableOp+sequential_5/dense_5/BiasAdd/ReadVariableOp2X
*sequential_5/dense_5/MatMul/ReadVariableOp*sequential_5/dense_5/MatMul/ReadVariableOp:J F
'
_output_shapes
:????????? 

_user_specified_nameX
?
?
I__inference_sequential_5_layer_call_and_return_conditional_losses_2764078
dense_5_input8
&dense_5_matmul_readvariableop_resource: 5
'dense_5_biasadd_readvariableop_resource: 
identity??dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMuldense_5_input%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_5/BiasAddy
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_5/Sigmoid?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_5/kernel/Regularizer/Square?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/Const?
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
IdentityIdentitydense_5/Sigmoid:y:0^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_5_input
?"
?
I__inference_sequential_4_layer_call_and_return_conditional_losses_2763281

inputs!
dense_4_2763260: 
dense_4_2763262:
identity

identity_1??dense_4/StatefulPartitionedCall?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
dense_4/StatefulPartitionedCallStatefulPartitionedCallinputsdense_4_2763260dense_4_2763262*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_27632592!
dense_4/StatefulPartitionedCall?
+dense_4/ActivityRegularizer/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *9
f4R2
0__inference_dense_4_activity_regularizer_27632352-
+dense_4/ActivityRegularizer/PartitionedCall?
!dense_4/ActivityRegularizer/ShapeShape(dense_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_4/ActivityRegularizer/Shape?
/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_4/ActivityRegularizer/strided_slice/stack?
1dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_4/ActivityRegularizer/strided_slice/stack_1?
1dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_4/ActivityRegularizer/strided_slice/stack_2?
)dense_4/ActivityRegularizer/strided_sliceStridedSlice*dense_4/ActivityRegularizer/Shape:output:08dense_4/ActivityRegularizer/strided_slice/stack:output:0:dense_4/ActivityRegularizer/strided_slice/stack_1:output:0:dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_4/ActivityRegularizer/strided_slice?
 dense_4/ActivityRegularizer/CastCast2dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_4/ActivityRegularizer/Cast?
#dense_4/ActivityRegularizer/truedivRealDiv4dense_4/ActivityRegularizer/PartitionedCall:output:0$dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_4/ActivityRegularizer/truediv?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_2763260*
_output_shapes

: *
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_4/kernel/Regularizer/Square?
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Const?
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum?
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_4/kernel/Regularizer/mul/x?
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul?
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity'dense_4/ActivityRegularizer/truediv:z:0 ^dense_4/StatefulPartitionedCall1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
)__inference_dense_4_layer_call_fn_2764140

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_27632592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
.__inference_sequential_4_layer_call_fn_2764004

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_27633472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
)__inference_dense_5_layer_call_fn_2764183

inputs
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_27634372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_dense_4_layer_call_and_return_conditional_losses_2763259

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_4/kernel/Regularizer/Square?
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Const?
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum?
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_4/kernel/Regularizer/mul/x?
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?A
?
I__inference_sequential_4_layer_call_and_return_conditional_losses_2763984

inputs8
&dense_4_matmul_readvariableop_resource: 5
'dense_4_biasadd_readvariableop_resource:
identity

identity_1??dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulinputs%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_4/BiasAddy
dense_4/SigmoidSigmoiddense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_4/Sigmoid?
#dense_4/ActivityRegularizer/SigmoidSigmoiddense_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2%
#dense_4/ActivityRegularizer/Sigmoid?
2dense_4/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 24
2dense_4/ActivityRegularizer/Mean/reduction_indices?
 dense_4/ActivityRegularizer/MeanMean'dense_4/ActivityRegularizer/Sigmoid:y:0;dense_4/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
:2"
 dense_4/ActivityRegularizer/Mean?
%dense_4/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2'
%dense_4/ActivityRegularizer/Maximum/y?
#dense_4/ActivityRegularizer/MaximumMaximum)dense_4/ActivityRegularizer/Mean:output:0.dense_4/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
:2%
#dense_4/ActivityRegularizer/Maximum?
%dense_4/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2'
%dense_4/ActivityRegularizer/truediv/x?
#dense_4/ActivityRegularizer/truedivRealDiv.dense_4/ActivityRegularizer/truediv/x:output:0'dense_4/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
:2%
#dense_4/ActivityRegularizer/truediv?
dense_4/ActivityRegularizer/LogLog'dense_4/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
:2!
dense_4/ActivityRegularizer/Log?
!dense_4/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_4/ActivityRegularizer/mul/x?
dense_4/ActivityRegularizer/mulMul*dense_4/ActivityRegularizer/mul/x:output:0#dense_4/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
:2!
dense_4/ActivityRegularizer/mul?
!dense_4/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!dense_4/ActivityRegularizer/sub/x?
dense_4/ActivityRegularizer/subSub*dense_4/ActivityRegularizer/sub/x:output:0'dense_4/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
:2!
dense_4/ActivityRegularizer/sub?
'dense_4/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2)
'dense_4/ActivityRegularizer/truediv_1/x?
%dense_4/ActivityRegularizer/truediv_1RealDiv0dense_4/ActivityRegularizer/truediv_1/x:output:0#dense_4/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
:2'
%dense_4/ActivityRegularizer/truediv_1?
!dense_4/ActivityRegularizer/Log_1Log)dense_4/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
:2#
!dense_4/ActivityRegularizer/Log_1?
#dense_4/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2%
#dense_4/ActivityRegularizer/mul_1/x?
!dense_4/ActivityRegularizer/mul_1Mul,dense_4/ActivityRegularizer/mul_1/x:output:0%dense_4/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
:2#
!dense_4/ActivityRegularizer/mul_1?
dense_4/ActivityRegularizer/addAddV2#dense_4/ActivityRegularizer/mul:z:0%dense_4/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
:2!
dense_4/ActivityRegularizer/add?
!dense_4/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_4/ActivityRegularizer/Const?
dense_4/ActivityRegularizer/SumSum#dense_4/ActivityRegularizer/add:z:0*dense_4/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_4/ActivityRegularizer/Sum?
#dense_4/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#dense_4/ActivityRegularizer/mul_2/x?
!dense_4/ActivityRegularizer/mul_2Mul,dense_4/ActivityRegularizer/mul_2/x:output:0(dense_4/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_4/ActivityRegularizer/mul_2?
!dense_4/ActivityRegularizer/ShapeShapedense_4/Sigmoid:y:0*
T0*
_output_shapes
:2#
!dense_4/ActivityRegularizer/Shape?
/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_4/ActivityRegularizer/strided_slice/stack?
1dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_4/ActivityRegularizer/strided_slice/stack_1?
1dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_4/ActivityRegularizer/strided_slice/stack_2?
)dense_4/ActivityRegularizer/strided_sliceStridedSlice*dense_4/ActivityRegularizer/Shape:output:08dense_4/ActivityRegularizer/strided_slice/stack:output:0:dense_4/ActivityRegularizer/strided_slice/stack_1:output:0:dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_4/ActivityRegularizer/strided_slice?
 dense_4/ActivityRegularizer/CastCast2dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_4/ActivityRegularizer/Cast?
%dense_4/ActivityRegularizer/truediv_2RealDiv%dense_4/ActivityRegularizer/mul_2:z:0$dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_4/ActivityRegularizer/truediv_2?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_4/kernel/Regularizer/Square?
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Const?
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum?
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_4/kernel/Regularizer/mul/x?
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul?
IdentityIdentitydense_4/Sigmoid:y:0^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity)dense_4/ActivityRegularizer/truediv_2:z:0^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
.__inference_sequential_5_layer_call_fn_2764114
dense_5_input
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_5_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_27634932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_5_input
?
?
__inference_loss_fn_1_2764194K
9dense_5_kernel_regularizer_square_readvariableop_resource: 
identity??0dense_5/kernel/Regularizer/Square/ReadVariableOp?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_5_kernel_regularizer_square_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_5/kernel/Regularizer/Square?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/Const?
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
IdentityIdentity"dense_5/kernel/Regularizer/mul:z:01^dense_5/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp
?
?
D__inference_dense_5_layer_call_and_return_conditional_losses_2763437

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2	
Sigmoid?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_5/kernel/Regularizer/Square?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/Const?
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?"
?
I__inference_sequential_4_layer_call_and_return_conditional_losses_2763389
input_3!
dense_4_2763368: 
dense_4_2763370:
identity

identity_1??dense_4/StatefulPartitionedCall?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
dense_4/StatefulPartitionedCallStatefulPartitionedCallinput_3dense_4_2763368dense_4_2763370*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_27632592!
dense_4/StatefulPartitionedCall?
+dense_4/ActivityRegularizer/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *9
f4R2
0__inference_dense_4_activity_regularizer_27632352-
+dense_4/ActivityRegularizer/PartitionedCall?
!dense_4/ActivityRegularizer/ShapeShape(dense_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_4/ActivityRegularizer/Shape?
/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_4/ActivityRegularizer/strided_slice/stack?
1dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_4/ActivityRegularizer/strided_slice/stack_1?
1dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_4/ActivityRegularizer/strided_slice/stack_2?
)dense_4/ActivityRegularizer/strided_sliceStridedSlice*dense_4/ActivityRegularizer/Shape:output:08dense_4/ActivityRegularizer/strided_slice/stack:output:0:dense_4/ActivityRegularizer/strided_slice/stack_1:output:0:dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_4/ActivityRegularizer/strided_slice?
 dense_4/ActivityRegularizer/CastCast2dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_4/ActivityRegularizer/Cast?
#dense_4/ActivityRegularizer/truedivRealDiv4dense_4/ActivityRegularizer/PartitionedCall:output:0$dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_4/ActivityRegularizer/truediv?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_2763368*
_output_shapes

: *
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_4/kernel/Regularizer/Square?
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Const?
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum?
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_4/kernel/Regularizer/mul/x?
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul?
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity'dense_4/ActivityRegularizer/truediv:z:0 ^dense_4/StatefulPartitionedCall1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:????????? 
!
_user_specified_name	input_3
?
?
.__inference_sequential_5_layer_call_fn_2764096

inputs
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_27634502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?\
?
"__inference__wrapped_model_2763205
input_1S
Aautoencoder_2_sequential_4_dense_4_matmul_readvariableop_resource: P
Bautoencoder_2_sequential_4_dense_4_biasadd_readvariableop_resource:S
Aautoencoder_2_sequential_5_dense_5_matmul_readvariableop_resource: P
Bautoencoder_2_sequential_5_dense_5_biasadd_readvariableop_resource: 
identity??9autoencoder_2/sequential_4/dense_4/BiasAdd/ReadVariableOp?8autoencoder_2/sequential_4/dense_4/MatMul/ReadVariableOp?9autoencoder_2/sequential_5/dense_5/BiasAdd/ReadVariableOp?8autoencoder_2/sequential_5/dense_5/MatMul/ReadVariableOp?
8autoencoder_2/sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOpAautoencoder_2_sequential_4_dense_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype02:
8autoencoder_2/sequential_4/dense_4/MatMul/ReadVariableOp?
)autoencoder_2/sequential_4/dense_4/MatMulMatMulinput_1@autoencoder_2/sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2+
)autoencoder_2/sequential_4/dense_4/MatMul?
9autoencoder_2/sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOpBautoencoder_2_sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9autoencoder_2/sequential_4/dense_4/BiasAdd/ReadVariableOp?
*autoencoder_2/sequential_4/dense_4/BiasAddBiasAdd3autoencoder_2/sequential_4/dense_4/MatMul:product:0Aautoencoder_2/sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2,
*autoencoder_2/sequential_4/dense_4/BiasAdd?
*autoencoder_2/sequential_4/dense_4/SigmoidSigmoid3autoencoder_2/sequential_4/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2,
*autoencoder_2/sequential_4/dense_4/Sigmoid?
>autoencoder_2/sequential_4/dense_4/ActivityRegularizer/SigmoidSigmoid.autoencoder_2/sequential_4/dense_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2@
>autoencoder_2/sequential_4/dense_4/ActivityRegularizer/Sigmoid?
Mautoencoder_2/sequential_4/dense_4/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mautoencoder_2/sequential_4/dense_4/ActivityRegularizer/Mean/reduction_indices?
;autoencoder_2/sequential_4/dense_4/ActivityRegularizer/MeanMeanBautoencoder_2/sequential_4/dense_4/ActivityRegularizer/Sigmoid:y:0Vautoencoder_2/sequential_4/dense_4/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
:2=
;autoencoder_2/sequential_4/dense_4/ActivityRegularizer/Mean?
@autoencoder_2/sequential_4/dense_4/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2B
@autoencoder_2/sequential_4/dense_4/ActivityRegularizer/Maximum/y?
>autoencoder_2/sequential_4/dense_4/ActivityRegularizer/MaximumMaximumDautoencoder_2/sequential_4/dense_4/ActivityRegularizer/Mean:output:0Iautoencoder_2/sequential_4/dense_4/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
:2@
>autoencoder_2/sequential_4/dense_4/ActivityRegularizer/Maximum?
@autoencoder_2/sequential_4/dense_4/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2B
@autoencoder_2/sequential_4/dense_4/ActivityRegularizer/truediv/x?
>autoencoder_2/sequential_4/dense_4/ActivityRegularizer/truedivRealDivIautoencoder_2/sequential_4/dense_4/ActivityRegularizer/truediv/x:output:0Bautoencoder_2/sequential_4/dense_4/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
:2@
>autoencoder_2/sequential_4/dense_4/ActivityRegularizer/truediv?
:autoencoder_2/sequential_4/dense_4/ActivityRegularizer/LogLogBautoencoder_2/sequential_4/dense_4/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
:2<
:autoencoder_2/sequential_4/dense_4/ActivityRegularizer/Log?
<autoencoder_2/sequential_4/dense_4/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2>
<autoencoder_2/sequential_4/dense_4/ActivityRegularizer/mul/x?
:autoencoder_2/sequential_4/dense_4/ActivityRegularizer/mulMulEautoencoder_2/sequential_4/dense_4/ActivityRegularizer/mul/x:output:0>autoencoder_2/sequential_4/dense_4/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
:2<
:autoencoder_2/sequential_4/dense_4/ActivityRegularizer/mul?
<autoencoder_2/sequential_4/dense_4/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2>
<autoencoder_2/sequential_4/dense_4/ActivityRegularizer/sub/x?
:autoencoder_2/sequential_4/dense_4/ActivityRegularizer/subSubEautoencoder_2/sequential_4/dense_4/ActivityRegularizer/sub/x:output:0Bautoencoder_2/sequential_4/dense_4/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
:2<
:autoencoder_2/sequential_4/dense_4/ActivityRegularizer/sub?
Bautoencoder_2/sequential_4/dense_4/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2D
Bautoencoder_2/sequential_4/dense_4/ActivityRegularizer/truediv_1/x?
@autoencoder_2/sequential_4/dense_4/ActivityRegularizer/truediv_1RealDivKautoencoder_2/sequential_4/dense_4/ActivityRegularizer/truediv_1/x:output:0>autoencoder_2/sequential_4/dense_4/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
:2B
@autoencoder_2/sequential_4/dense_4/ActivityRegularizer/truediv_1?
<autoencoder_2/sequential_4/dense_4/ActivityRegularizer/Log_1LogDautoencoder_2/sequential_4/dense_4/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
:2>
<autoencoder_2/sequential_4/dense_4/ActivityRegularizer/Log_1?
>autoencoder_2/sequential_4/dense_4/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2@
>autoencoder_2/sequential_4/dense_4/ActivityRegularizer/mul_1/x?
<autoencoder_2/sequential_4/dense_4/ActivityRegularizer/mul_1MulGautoencoder_2/sequential_4/dense_4/ActivityRegularizer/mul_1/x:output:0@autoencoder_2/sequential_4/dense_4/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
:2>
<autoencoder_2/sequential_4/dense_4/ActivityRegularizer/mul_1?
:autoencoder_2/sequential_4/dense_4/ActivityRegularizer/addAddV2>autoencoder_2/sequential_4/dense_4/ActivityRegularizer/mul:z:0@autoencoder_2/sequential_4/dense_4/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
:2<
:autoencoder_2/sequential_4/dense_4/ActivityRegularizer/add?
<autoencoder_2/sequential_4/dense_4/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2>
<autoencoder_2/sequential_4/dense_4/ActivityRegularizer/Const?
:autoencoder_2/sequential_4/dense_4/ActivityRegularizer/SumSum>autoencoder_2/sequential_4/dense_4/ActivityRegularizer/add:z:0Eautoencoder_2/sequential_4/dense_4/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2<
:autoencoder_2/sequential_4/dense_4/ActivityRegularizer/Sum?
>autoencoder_2/sequential_4/dense_4/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2@
>autoencoder_2/sequential_4/dense_4/ActivityRegularizer/mul_2/x?
<autoencoder_2/sequential_4/dense_4/ActivityRegularizer/mul_2MulGautoencoder_2/sequential_4/dense_4/ActivityRegularizer/mul_2/x:output:0Cautoencoder_2/sequential_4/dense_4/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2>
<autoencoder_2/sequential_4/dense_4/ActivityRegularizer/mul_2?
<autoencoder_2/sequential_4/dense_4/ActivityRegularizer/ShapeShape.autoencoder_2/sequential_4/dense_4/Sigmoid:y:0*
T0*
_output_shapes
:2>
<autoencoder_2/sequential_4/dense_4/ActivityRegularizer/Shape?
Jautoencoder_2/sequential_4/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2L
Jautoencoder_2/sequential_4/dense_4/ActivityRegularizer/strided_slice/stack?
Lautoencoder_2/sequential_4/dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2N
Lautoencoder_2/sequential_4/dense_4/ActivityRegularizer/strided_slice/stack_1?
Lautoencoder_2/sequential_4/dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2N
Lautoencoder_2/sequential_4/dense_4/ActivityRegularizer/strided_slice/stack_2?
Dautoencoder_2/sequential_4/dense_4/ActivityRegularizer/strided_sliceStridedSliceEautoencoder_2/sequential_4/dense_4/ActivityRegularizer/Shape:output:0Sautoencoder_2/sequential_4/dense_4/ActivityRegularizer/strided_slice/stack:output:0Uautoencoder_2/sequential_4/dense_4/ActivityRegularizer/strided_slice/stack_1:output:0Uautoencoder_2/sequential_4/dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2F
Dautoencoder_2/sequential_4/dense_4/ActivityRegularizer/strided_slice?
;autoencoder_2/sequential_4/dense_4/ActivityRegularizer/CastCastMautoencoder_2/sequential_4/dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2=
;autoencoder_2/sequential_4/dense_4/ActivityRegularizer/Cast?
@autoencoder_2/sequential_4/dense_4/ActivityRegularizer/truediv_2RealDiv@autoencoder_2/sequential_4/dense_4/ActivityRegularizer/mul_2:z:0?autoencoder_2/sequential_4/dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2B
@autoencoder_2/sequential_4/dense_4/ActivityRegularizer/truediv_2?
8autoencoder_2/sequential_5/dense_5/MatMul/ReadVariableOpReadVariableOpAautoencoder_2_sequential_5_dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02:
8autoencoder_2/sequential_5/dense_5/MatMul/ReadVariableOp?
)autoencoder_2/sequential_5/dense_5/MatMulMatMul.autoencoder_2/sequential_4/dense_4/Sigmoid:y:0@autoencoder_2/sequential_5/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2+
)autoencoder_2/sequential_5/dense_5/MatMul?
9autoencoder_2/sequential_5/dense_5/BiasAdd/ReadVariableOpReadVariableOpBautoencoder_2_sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02;
9autoencoder_2/sequential_5/dense_5/BiasAdd/ReadVariableOp?
*autoencoder_2/sequential_5/dense_5/BiasAddBiasAdd3autoencoder_2/sequential_5/dense_5/MatMul:product:0Aautoencoder_2/sequential_5/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2,
*autoencoder_2/sequential_5/dense_5/BiasAdd?
*autoencoder_2/sequential_5/dense_5/SigmoidSigmoid3autoencoder_2/sequential_5/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2,
*autoencoder_2/sequential_5/dense_5/Sigmoid?
IdentityIdentity.autoencoder_2/sequential_5/dense_5/Sigmoid:y:0:^autoencoder_2/sequential_4/dense_4/BiasAdd/ReadVariableOp9^autoencoder_2/sequential_4/dense_4/MatMul/ReadVariableOp:^autoencoder_2/sequential_5/dense_5/BiasAdd/ReadVariableOp9^autoencoder_2/sequential_5/dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : : : 2v
9autoencoder_2/sequential_4/dense_4/BiasAdd/ReadVariableOp9autoencoder_2/sequential_4/dense_4/BiasAdd/ReadVariableOp2t
8autoencoder_2/sequential_4/dense_4/MatMul/ReadVariableOp8autoencoder_2/sequential_4/dense_4/MatMul/ReadVariableOp2v
9autoencoder_2/sequential_5/dense_5/BiasAdd/ReadVariableOp9autoencoder_2/sequential_5/dense_5/BiasAdd/ReadVariableOp2t
8autoencoder_2/sequential_5/dense_5/MatMul/ReadVariableOp8autoencoder_2/sequential_5/dense_5/MatMul/ReadVariableOp:P L
'
_output_shapes
:????????? 
!
_user_specified_name	input_1
?"
?
I__inference_sequential_4_layer_call_and_return_conditional_losses_2763347

inputs!
dense_4_2763326: 
dense_4_2763328:
identity

identity_1??dense_4/StatefulPartitionedCall?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
dense_4/StatefulPartitionedCallStatefulPartitionedCallinputsdense_4_2763326dense_4_2763328*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_27632592!
dense_4/StatefulPartitionedCall?
+dense_4/ActivityRegularizer/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *9
f4R2
0__inference_dense_4_activity_regularizer_27632352-
+dense_4/ActivityRegularizer/PartitionedCall?
!dense_4/ActivityRegularizer/ShapeShape(dense_4/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_4/ActivityRegularizer/Shape?
/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_4/ActivityRegularizer/strided_slice/stack?
1dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_4/ActivityRegularizer/strided_slice/stack_1?
1dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_4/ActivityRegularizer/strided_slice/stack_2?
)dense_4/ActivityRegularizer/strided_sliceStridedSlice*dense_4/ActivityRegularizer/Shape:output:08dense_4/ActivityRegularizer/strided_slice/stack:output:0:dense_4/ActivityRegularizer/strided_slice/stack_1:output:0:dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_4/ActivityRegularizer/strided_slice?
 dense_4/ActivityRegularizer/CastCast2dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_4/ActivityRegularizer/Cast?
#dense_4/ActivityRegularizer/truedivRealDiv4dense_4/ActivityRegularizer/PartitionedCall:output:0$dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_4/ActivityRegularizer/truediv?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_4_2763326*
_output_shapes

: *
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_4/kernel/Regularizer/Square?
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Const?
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum?
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_4/kernel/Regularizer/mul/x?
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul?
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity'dense_4/ActivityRegularizer/truediv:z:0 ^dense_4/StatefulPartitionedCall1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
/__inference_autoencoder_2_layer_call_fn_2763583
input_1
unknown: 
	unknown_0:
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_autoencoder_2_layer_call_and_return_conditional_losses_27635712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:????????? 
!
_user_specified_name	input_1
?d
?
J__inference_autoencoder_2_layer_call_and_return_conditional_losses_2763856
xE
3sequential_4_dense_4_matmul_readvariableop_resource: B
4sequential_4_dense_4_biasadd_readvariableop_resource:E
3sequential_5_dense_5_matmul_readvariableop_resource: B
4sequential_5_dense_5_biasadd_readvariableop_resource: 
identity

identity_1??0dense_4/kernel/Regularizer/Square/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?+sequential_4/dense_4/BiasAdd/ReadVariableOp?*sequential_4/dense_4/MatMul/ReadVariableOp?+sequential_5/dense_5/BiasAdd/ReadVariableOp?*sequential_5/dense_5/MatMul/ReadVariableOp?
*sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype02,
*sequential_4/dense_4/MatMul/ReadVariableOp?
sequential_4/dense_4/MatMulMatMulx2sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_4/dense_4/MatMul?
+sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_4/dense_4/BiasAdd/ReadVariableOp?
sequential_4/dense_4/BiasAddBiasAdd%sequential_4/dense_4/MatMul:product:03sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_4/dense_4/BiasAdd?
sequential_4/dense_4/SigmoidSigmoid%sequential_4/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_4/dense_4/Sigmoid?
0sequential_4/dense_4/ActivityRegularizer/SigmoidSigmoid sequential_4/dense_4/Sigmoid:y:0*
T0*'
_output_shapes
:?????????22
0sequential_4/dense_4/ActivityRegularizer/Sigmoid?
?sequential_4/dense_4/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2A
?sequential_4/dense_4/ActivityRegularizer/Mean/reduction_indices?
-sequential_4/dense_4/ActivityRegularizer/MeanMean4sequential_4/dense_4/ActivityRegularizer/Sigmoid:y:0Hsequential_4/dense_4/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
:2/
-sequential_4/dense_4/ActivityRegularizer/Mean?
2sequential_4/dense_4/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.24
2sequential_4/dense_4/ActivityRegularizer/Maximum/y?
0sequential_4/dense_4/ActivityRegularizer/MaximumMaximum6sequential_4/dense_4/ActivityRegularizer/Mean:output:0;sequential_4/dense_4/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
:22
0sequential_4/dense_4/ActivityRegularizer/Maximum?
2sequential_4/dense_4/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<24
2sequential_4/dense_4/ActivityRegularizer/truediv/x?
0sequential_4/dense_4/ActivityRegularizer/truedivRealDiv;sequential_4/dense_4/ActivityRegularizer/truediv/x:output:04sequential_4/dense_4/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
:22
0sequential_4/dense_4/ActivityRegularizer/truediv?
,sequential_4/dense_4/ActivityRegularizer/LogLog4sequential_4/dense_4/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
:2.
,sequential_4/dense_4/ActivityRegularizer/Log?
.sequential_4/dense_4/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.sequential_4/dense_4/ActivityRegularizer/mul/x?
,sequential_4/dense_4/ActivityRegularizer/mulMul7sequential_4/dense_4/ActivityRegularizer/mul/x:output:00sequential_4/dense_4/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
:2.
,sequential_4/dense_4/ActivityRegularizer/mul?
.sequential_4/dense_4/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??20
.sequential_4/dense_4/ActivityRegularizer/sub/x?
,sequential_4/dense_4/ActivityRegularizer/subSub7sequential_4/dense_4/ActivityRegularizer/sub/x:output:04sequential_4/dense_4/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
:2.
,sequential_4/dense_4/ActivityRegularizer/sub?
4sequential_4/dense_4/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?26
4sequential_4/dense_4/ActivityRegularizer/truediv_1/x?
2sequential_4/dense_4/ActivityRegularizer/truediv_1RealDiv=sequential_4/dense_4/ActivityRegularizer/truediv_1/x:output:00sequential_4/dense_4/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
:24
2sequential_4/dense_4/ActivityRegularizer/truediv_1?
.sequential_4/dense_4/ActivityRegularizer/Log_1Log6sequential_4/dense_4/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
:20
.sequential_4/dense_4/ActivityRegularizer/Log_1?
0sequential_4/dense_4/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?22
0sequential_4/dense_4/ActivityRegularizer/mul_1/x?
.sequential_4/dense_4/ActivityRegularizer/mul_1Mul9sequential_4/dense_4/ActivityRegularizer/mul_1/x:output:02sequential_4/dense_4/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
:20
.sequential_4/dense_4/ActivityRegularizer/mul_1?
,sequential_4/dense_4/ActivityRegularizer/addAddV20sequential_4/dense_4/ActivityRegularizer/mul:z:02sequential_4/dense_4/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
:2.
,sequential_4/dense_4/ActivityRegularizer/add?
.sequential_4/dense_4/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_4/dense_4/ActivityRegularizer/Const?
,sequential_4/dense_4/ActivityRegularizer/SumSum0sequential_4/dense_4/ActivityRegularizer/add:z:07sequential_4/dense_4/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2.
,sequential_4/dense_4/ActivityRegularizer/Sum?
0sequential_4/dense_4/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??22
0sequential_4/dense_4/ActivityRegularizer/mul_2/x?
.sequential_4/dense_4/ActivityRegularizer/mul_2Mul9sequential_4/dense_4/ActivityRegularizer/mul_2/x:output:05sequential_4/dense_4/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 20
.sequential_4/dense_4/ActivityRegularizer/mul_2?
.sequential_4/dense_4/ActivityRegularizer/ShapeShape sequential_4/dense_4/Sigmoid:y:0*
T0*
_output_shapes
:20
.sequential_4/dense_4/ActivityRegularizer/Shape?
<sequential_4/dense_4/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential_4/dense_4/ActivityRegularizer/strided_slice/stack?
>sequential_4/dense_4/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_4/dense_4/ActivityRegularizer/strided_slice/stack_1?
>sequential_4/dense_4/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_4/dense_4/ActivityRegularizer/strided_slice/stack_2?
6sequential_4/dense_4/ActivityRegularizer/strided_sliceStridedSlice7sequential_4/dense_4/ActivityRegularizer/Shape:output:0Esequential_4/dense_4/ActivityRegularizer/strided_slice/stack:output:0Gsequential_4/dense_4/ActivityRegularizer/strided_slice/stack_1:output:0Gsequential_4/dense_4/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6sequential_4/dense_4/ActivityRegularizer/strided_slice?
-sequential_4/dense_4/ActivityRegularizer/CastCast?sequential_4/dense_4/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-sequential_4/dense_4/ActivityRegularizer/Cast?
2sequential_4/dense_4/ActivityRegularizer/truediv_2RealDiv2sequential_4/dense_4/ActivityRegularizer/mul_2:z:01sequential_4/dense_4/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 24
2sequential_4/dense_4/ActivityRegularizer/truediv_2?
*sequential_5/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_5_dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02,
*sequential_5/dense_5/MatMul/ReadVariableOp?
sequential_5/dense_5/MatMulMatMul sequential_4/dense_4/Sigmoid:y:02sequential_5/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_5/dense_5/MatMul?
+sequential_5/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+sequential_5/dense_5/BiasAdd/ReadVariableOp?
sequential_5/dense_5/BiasAddBiasAdd%sequential_5/dense_5/MatMul:product:03sequential_5/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_5/dense_5/BiasAdd?
sequential_5/dense_5/SigmoidSigmoid%sequential_5/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
sequential_5/dense_5/Sigmoid?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_4/kernel/Regularizer/Square?
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Const?
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum?
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_4/kernel/Regularizer/mul/x?
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_5_dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_5/kernel/Regularizer/Square?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/Const?
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
IdentityIdentity sequential_5/dense_5/Sigmoid:y:01^dense_4/kernel/Regularizer/Square/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp,^sequential_4/dense_4/BiasAdd/ReadVariableOp+^sequential_4/dense_4/MatMul/ReadVariableOp,^sequential_5/dense_5/BiasAdd/ReadVariableOp+^sequential_5/dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity6sequential_4/dense_4/ActivityRegularizer/truediv_2:z:01^dense_4/kernel/Regularizer/Square/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp,^sequential_4/dense_4/BiasAdd/ReadVariableOp+^sequential_4/dense_4/MatMul/ReadVariableOp,^sequential_5/dense_5/BiasAdd/ReadVariableOp+^sequential_5/dense_5/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : : : 2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp2Z
+sequential_4/dense_4/BiasAdd/ReadVariableOp+sequential_4/dense_4/BiasAdd/ReadVariableOp2X
*sequential_4/dense_4/MatMul/ReadVariableOp*sequential_4/dense_4/MatMul/ReadVariableOp2Z
+sequential_5/dense_5/BiasAdd/ReadVariableOp+sequential_5/dense_5/BiasAdd/ReadVariableOp2X
*sequential_5/dense_5/MatMul/ReadVariableOp*sequential_5/dense_5/MatMul/ReadVariableOp:J F
'
_output_shapes
:????????? 

_user_specified_nameX
?
?
D__inference_dense_4_layer_call_and_return_conditional_losses_2764211

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_4/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_4/kernel/Regularizer/Square?
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Const?
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum?
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_4/kernel/Regularizer/mul/x?
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_4/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
I__inference_sequential_5_layer_call_and_return_conditional_losses_2764061
dense_5_input8
&dense_5_matmul_readvariableop_resource: 5
'dense_5_biasadd_readvariableop_resource: 
identity??dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMuldense_5_input%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_5/BiasAddy
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_5/Sigmoid?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_5/kernel/Regularizer/Square?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/Const?
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
IdentityIdentitydense_5/Sigmoid:y:0^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_5_input
?
?
.__inference_sequential_5_layer_call_fn_2764087
dense_5_input
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_5_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_27634502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namedense_5_input
?
?
 __inference__traced_save_2764246
file_prefix-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*7
_input_shapes&
$: : :: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :

_output_shapes
: 
?
?
#__inference__traced_restore_2764268
file_prefix1
assignvariableop_dense_4_kernel: -
assignvariableop_1_dense_4_bias:3
!assignvariableop_2_dense_5_kernel: -
assignvariableop_3_dense_5_bias: 

identity_5??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_5_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_5_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_3:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?$
?
J__inference_autoencoder_2_layer_call_and_return_conditional_losses_2763571
x&
sequential_4_2763546: "
sequential_4_2763548:&
sequential_5_2763552: "
sequential_5_2763554: 
identity

identity_1??0dense_4/kernel/Regularizer/Square/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?$sequential_4/StatefulPartitionedCall?$sequential_5/StatefulPartitionedCall?
$sequential_4/StatefulPartitionedCallStatefulPartitionedCallxsequential_4_2763546sequential_4_2763548*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_27632812&
$sequential_4/StatefulPartitionedCall?
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall-sequential_4/StatefulPartitionedCall:output:0sequential_5_2763552sequential_5_2763554*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_27634502&
$sequential_5/StatefulPartitionedCall?
0dense_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_4_2763546*
_output_shapes

: *
dtype022
0dense_4/kernel/Regularizer/Square/ReadVariableOp?
!dense_4/kernel/Regularizer/SquareSquare8dense_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_4/kernel/Regularizer/Square?
 dense_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_4/kernel/Regularizer/Const?
dense_4/kernel/Regularizer/SumSum%dense_4/kernel/Regularizer/Square:y:0)dense_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/Sum?
 dense_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_4/kernel/Regularizer/mul/x?
dense_4/kernel/Regularizer/mulMul)dense_4/kernel/Regularizer/mul/x:output:0'dense_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_4/kernel/Regularizer/mul?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_5_2763552*
_output_shapes

: *
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_5/kernel/Regularizer/Square?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/Const?
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
IdentityIdentity-sequential_5/StatefulPartitionedCall:output:01^dense_4/kernel/Regularizer/Square/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity-sequential_4/StatefulPartitionedCall:output:11^dense_4/kernel/Regularizer/Square/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : : : 2d
0dense_4/kernel/Regularizer/Square/ReadVariableOp0dense_4/kernel/Regularizer/Square/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall:J F
'
_output_shapes
:????????? 

_user_specified_nameX
?
?
D__inference_dense_5_layer_call_and_return_conditional_losses_2764174

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2	
Sigmoid?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_5/kernel/Regularizer/Square?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/Const?
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_5_layer_call_fn_2764105

inputs
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_5_layer_call_and_return_conditional_losses_27634932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_5_layer_call_and_return_conditional_losses_2763493

inputs!
dense_5_2763481: 
dense_5_2763483: 
identity??dense_5/StatefulPartitionedCall?0dense_5/kernel/Regularizer/Square/ReadVariableOp?
dense_5/StatefulPartitionedCallStatefulPartitionedCallinputsdense_5_2763481dense_5_2763483*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_27634372!
dense_5/StatefulPartitionedCall?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_2763481*
_output_shapes

: *
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_5/kernel/Regularizer/Square?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/Const?
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_5/StatefulPartitionedCall1^dense_5/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_4_layer_call_fn_2763365
input_3
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_4_layer_call_and_return_conditional_losses_27633472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:????????? 
!
_user_specified_name	input_3
?
?
/__inference_autoencoder_2_layer_call_fn_2763884
x
unknown: 
	unknown_0:
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_autoencoder_2_layer_call_and_return_conditional_losses_27636272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:????????? 

_user_specified_nameX
?
?
I__inference_sequential_5_layer_call_and_return_conditional_losses_2764027

inputs8
&dense_5_matmul_readvariableop_resource: 5
'dense_5_biasadd_readvariableop_resource: 
identity??dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_5/BiasAddy
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_5/Sigmoid?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: 2#
!dense_5/kernel/Regularizer/Square?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_5/kernel/Regularizer/Const?
dense_5/kernel/Regularizer/SumSum%dense_5/kernel/Regularizer/Square:y:0)dense_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
IdentityIdentitydense_5/Sigmoid:y:0^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
H__inference_dense_4_layer_call_and_return_all_conditional_losses_2764131

inputs
unknown: 
	unknown_0:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_27632592
StatefulPartitionedCall?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *9
f4R2
0__inference_dense_4_activity_regularizer_27632352
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_2763736
input_1
unknown: 
	unknown_0:
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_27632052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:????????? 
!
_user_specified_name	input_1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0????????? <
output_10
StatefulPartitionedCall:0????????? tensorflow/serving/predict:ܰ
?
history
encoder
decoder
	variables
trainable_variables
regularization_losses
	keras_api

signatures
*8&call_and_return_all_conditional_losses
9_default_save_signature
:__call__"?
_tf_keras_model?{"name": "autoencoder_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Autoencoder", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 32]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Autoencoder"}}
 "
trackable_dict_wrapper
?
	layer_with_weights-0
	layer-0

	variables
trainable_variables
regularization_losses
	keras_api
*;&call_and_return_all_conditional_losses
<__call__"?
_tf_keras_sequential?{"name": "sequential_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 16, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}}, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 6}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 32]}, "float32", "input_3"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 16, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}]}}}
?
layer_with_weights-0
layer-0
	variables
trainable_variables
regularization_losses
	keras_api
*=&call_and_return_all_conditional_losses
>__call__"?
_tf_keras_sequential?{"name": "sequential_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_5_input"}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [105, 16]}, "float32", "dense_5_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_5_input"}, "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8}]}}}
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
non_trainable_variables
trainable_variables
metrics
layer_metrics
regularization_losses

layers
layer_regularization_losses
:__call__
9_default_save_signature
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
,
?serving_default"
signature_map
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*@&call_and_return_all_conditional_losses
A__call__"?

_tf_keras_layer?	{"name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 16, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 6}, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
B0"
trackable_list_wrapper
?

	variables
 non_trainable_variables
trainable_variables
!metrics
"layer_metrics
regularization_losses

#layers
$layer_regularization_losses
<__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
?	

kernel
bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
*C&call_and_return_all_conditional_losses
D__call__"?
_tf_keras_layer?{"name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [105, 16]}}
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
E0"
trackable_list_wrapper
?
	variables
)non_trainable_variables
trainable_variables
*metrics
+layer_metrics
regularization_losses

,layers
-layer_regularization_losses
>__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
 : 2dense_4/kernel
:2dense_4/bias
 : 2dense_5/kernel
: 2dense_5/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
B0"
trackable_list_wrapper
?
	variables
.non_trainable_variables
trainable_variables
/metrics
0layer_metrics
regularization_losses

1layers
2layer_regularization_losses
A__call__
Factivity_regularizer_fn
*@&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
	0"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
E0"
trackable_list_wrapper
?
%	variables
3non_trainable_variables
&trainable_variables
4metrics
5layer_metrics
'regularization_losses

6layers
7layer_regularization_losses
D__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
B0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
E0"
trackable_list_wrapper
?2?
J__inference_autoencoder_2_layer_call_and_return_conditional_losses_2763796
J__inference_autoencoder_2_layer_call_and_return_conditional_losses_2763856
J__inference_autoencoder_2_layer_call_and_return_conditional_losses_2763681
J__inference_autoencoder_2_layer_call_and_return_conditional_losses_2763709?
???
FullArgSpec$
args?
jself
jX

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
"__inference__wrapped_model_2763205?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1????????? 
?2?
/__inference_autoencoder_2_layer_call_fn_2763583
/__inference_autoencoder_2_layer_call_fn_2763870
/__inference_autoencoder_2_layer_call_fn_2763884
/__inference_autoencoder_2_layer_call_fn_2763653?
???
FullArgSpec$
args?
jself
jX

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_sequential_4_layer_call_and_return_conditional_losses_2763937
I__inference_sequential_4_layer_call_and_return_conditional_losses_2763984
I__inference_sequential_4_layer_call_and_return_conditional_losses_2763389
I__inference_sequential_4_layer_call_and_return_conditional_losses_2763413?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_sequential_4_layer_call_fn_2763289
.__inference_sequential_4_layer_call_fn_2763994
.__inference_sequential_4_layer_call_fn_2764004
.__inference_sequential_4_layer_call_fn_2763365?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_sequential_5_layer_call_and_return_conditional_losses_2764027
I__inference_sequential_5_layer_call_and_return_conditional_losses_2764044
I__inference_sequential_5_layer_call_and_return_conditional_losses_2764061
I__inference_sequential_5_layer_call_and_return_conditional_losses_2764078?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_sequential_5_layer_call_fn_2764087
.__inference_sequential_5_layer_call_fn_2764096
.__inference_sequential_5_layer_call_fn_2764105
.__inference_sequential_5_layer_call_fn_2764114?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
%__inference_signature_wrapper_2763736input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dense_4_layer_call_and_return_all_conditional_losses_2764131?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_4_layer_call_fn_2764140?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_2764151?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
D__inference_dense_5_layer_call_and_return_conditional_losses_2764174?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_5_layer_call_fn_2764183?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_1_2764194?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
0__inference_dense_4_activity_regularizer_2763235?
???
FullArgSpec!
args?
jself
j
activation
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
D__inference_dense_4_layer_call_and_return_conditional_losses_2764211?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_2763205m0?-
&?#
!?
input_1????????? 
? "3?0
.
output_1"?
output_1????????? ?
J__inference_autoencoder_2_layer_call_and_return_conditional_losses_2763681q4?1
*?'
!?
input_1????????? 
p 
? "3?0
?
0????????? 
?
?	
1/0 ?
J__inference_autoencoder_2_layer_call_and_return_conditional_losses_2763709q4?1
*?'
!?
input_1????????? 
p
? "3?0
?
0????????? 
?
?	
1/0 ?
J__inference_autoencoder_2_layer_call_and_return_conditional_losses_2763796k.?+
$?!
?
X????????? 
p 
? "3?0
?
0????????? 
?
?	
1/0 ?
J__inference_autoencoder_2_layer_call_and_return_conditional_losses_2763856k.?+
$?!
?
X????????? 
p
? "3?0
?
0????????? 
?
?	
1/0 ?
/__inference_autoencoder_2_layer_call_fn_2763583V4?1
*?'
!?
input_1????????? 
p 
? "?????????? ?
/__inference_autoencoder_2_layer_call_fn_2763653V4?1
*?'
!?
input_1????????? 
p
? "?????????? ?
/__inference_autoencoder_2_layer_call_fn_2763870P.?+
$?!
?
X????????? 
p 
? "?????????? ?
/__inference_autoencoder_2_layer_call_fn_2763884P.?+
$?!
?
X????????? 
p
? "?????????? c
0__inference_dense_4_activity_regularizer_2763235/$?!
?
?

activation
? "? ?
H__inference_dense_4_layer_call_and_return_all_conditional_losses_2764131j/?,
%?"
 ?
inputs????????? 
? "3?0
?
0?????????
?
?	
1/0 ?
D__inference_dense_4_layer_call_and_return_conditional_losses_2764211\/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? |
)__inference_dense_4_layer_call_fn_2764140O/?,
%?"
 ?
inputs????????? 
? "???????????
D__inference_dense_5_layer_call_and_return_conditional_losses_2764174\/?,
%?"
 ?
inputs?????????
? "%?"
?
0????????? 
? |
)__inference_dense_5_layer_call_fn_2764183O/?,
%?"
 ?
inputs?????????
? "?????????? <
__inference_loss_fn_0_2764151?

? 
? "? <
__inference_loss_fn_1_2764194?

? 
? "? ?
I__inference_sequential_4_layer_call_and_return_conditional_losses_2763389s8?5
.?+
!?
input_3????????? 
p 

 
? "3?0
?
0?????????
?
?	
1/0 ?
I__inference_sequential_4_layer_call_and_return_conditional_losses_2763413s8?5
.?+
!?
input_3????????? 
p

 
? "3?0
?
0?????????
?
?	
1/0 ?
I__inference_sequential_4_layer_call_and_return_conditional_losses_2763937r7?4
-?*
 ?
inputs????????? 
p 

 
? "3?0
?
0?????????
?
?	
1/0 ?
I__inference_sequential_4_layer_call_and_return_conditional_losses_2763984r7?4
-?*
 ?
inputs????????? 
p

 
? "3?0
?
0?????????
?
?	
1/0 ?
.__inference_sequential_4_layer_call_fn_2763289X8?5
.?+
!?
input_3????????? 
p 

 
? "???????????
.__inference_sequential_4_layer_call_fn_2763365X8?5
.?+
!?
input_3????????? 
p

 
? "???????????
.__inference_sequential_4_layer_call_fn_2763994W7?4
-?*
 ?
inputs????????? 
p 

 
? "???????????
.__inference_sequential_4_layer_call_fn_2764004W7?4
-?*
 ?
inputs????????? 
p

 
? "???????????
I__inference_sequential_5_layer_call_and_return_conditional_losses_2764027d7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0????????? 
? ?
I__inference_sequential_5_layer_call_and_return_conditional_losses_2764044d7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0????????? 
? ?
I__inference_sequential_5_layer_call_and_return_conditional_losses_2764061k>?;
4?1
'?$
dense_5_input?????????
p 

 
? "%?"
?
0????????? 
? ?
I__inference_sequential_5_layer_call_and_return_conditional_losses_2764078k>?;
4?1
'?$
dense_5_input?????????
p

 
? "%?"
?
0????????? 
? ?
.__inference_sequential_5_layer_call_fn_2764087^>?;
4?1
'?$
dense_5_input?????????
p 

 
? "?????????? ?
.__inference_sequential_5_layer_call_fn_2764096W7?4
-?*
 ?
inputs?????????
p 

 
? "?????????? ?
.__inference_sequential_5_layer_call_fn_2764105W7?4
-?*
 ?
inputs?????????
p

 
? "?????????? ?
.__inference_sequential_5_layer_call_fn_2764114^>?;
4?1
'?$
dense_5_input?????????
p

 
? "?????????? ?
%__inference_signature_wrapper_2763736x;?8
? 
1?.
,
input_1!?
input_1????????? "3?0
.
output_1"?
output_1????????? 