��

��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
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
�
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
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
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
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718��
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:s@*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:s@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@s*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@s*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:s*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:s*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
history
encoder
decoder
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
y
	layer_with_weights-0
	layer-0

trainable_variables
regularization_losses
	variables
	keras_api
y
layer_with_weights-0
layer-0
trainable_variables
regularization_losses
	variables
	keras_api

0
1
2
3
 

0
1
2
3
�
trainable_variables
layer_metrics
non_trainable_variables
layer_regularization_losses
regularization_losses
metrics
	variables

layers
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api

0
1
 

0
1
�

trainable_variables
 layer_metrics
!non_trainable_variables
"layer_regularization_losses
regularization_losses
#metrics
	variables

$layers
h

kernel
bias
%trainable_variables
&	variables
'regularization_losses
(	keras_api

0
1
 

0
1
�
trainable_variables
)layer_metrics
*non_trainable_variables
+layer_regularization_losses
regularization_losses
,metrics
	variables

-layers
RP
VARIABLE_VALUEdense/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUE
dense/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_1/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 

0
1

0
1

0
1
 
�
trainable_variables
.layer_metrics
/non_trainable_variables
	variables
0layer_regularization_losses
regularization_losses
1metrics

2layers
 
 
 
 

	0

0
1

0
1
 
�
%trainable_variables
3layer_metrics
4non_trainable_variables
&	variables
5layer_regularization_losses
'regularization_losses
6metrics

7layers
 
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
z
serving_default_input_1Placeholder*'
_output_shapes
:���������s*
dtype0*
shape:���������s
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������s*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_920850
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpConst*
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
GPU 2J 8� *(
f#R!
__inference__traced_save_921360
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/bias*
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
GPU 2J 8� *+
f&R$
"__inference__traced_restore_921382��
�
�
&__inference_dense_layer_call_fn_921243

inputs
unknown:s@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_9203732
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������s: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������s
 
_user_specified_nameinputs
�!
�
F__inference_sequential_layer_call_and_return_conditional_losses_920503
input_1
dense_920482:s@
dense_920484:@
identity

identity_1��dense/StatefulPartitionedCall�.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_920482dense_920484*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_9203732
dense/StatefulPartitionedCall�
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *6
f1R/
-__inference_dense_activity_regularizer_9203492+
)dense/ActivityRegularizer/PartitionedCall�
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2!
dense/ActivityRegularizer/Shape�
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense/ActivityRegularizer/strided_slice/stack�
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_1�
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_2�
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense/ActivityRegularizer/strided_slice�
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
dense/ActivityRegularizer/Cast�
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2#
!dense/ActivityRegularizer/truediv�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_920482*
_output_shapes

:s@*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:s@2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������s: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:���������s
!
_user_specified_name	input_1
�
�
-__inference_sequential_1_layer_call_fn_921160
dense_1_input
unknown:@s
	unknown_0:s
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������s*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_9206072
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������s2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:���������@
'
_user_specified_namedense_1_input
�V
�
!__inference__wrapped_model_920319
input_1M
;autoencoder_sequential_dense_matmul_readvariableop_resource:s@J
<autoencoder_sequential_dense_biasadd_readvariableop_resource:@Q
?autoencoder_sequential_1_dense_1_matmul_readvariableop_resource:@sN
@autoencoder_sequential_1_dense_1_biasadd_readvariableop_resource:s
identity��3autoencoder/sequential/dense/BiasAdd/ReadVariableOp�2autoencoder/sequential/dense/MatMul/ReadVariableOp�7autoencoder/sequential_1/dense_1/BiasAdd/ReadVariableOp�6autoencoder/sequential_1/dense_1/MatMul/ReadVariableOp�
2autoencoder/sequential/dense/MatMul/ReadVariableOpReadVariableOp;autoencoder_sequential_dense_matmul_readvariableop_resource*
_output_shapes

:s@*
dtype024
2autoencoder/sequential/dense/MatMul/ReadVariableOp�
#autoencoder/sequential/dense/MatMulMatMulinput_1:autoencoder/sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2%
#autoencoder/sequential/dense/MatMul�
3autoencoder/sequential/dense/BiasAdd/ReadVariableOpReadVariableOp<autoencoder_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype025
3autoencoder/sequential/dense/BiasAdd/ReadVariableOp�
$autoencoder/sequential/dense/BiasAddBiasAdd-autoencoder/sequential/dense/MatMul:product:0;autoencoder/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2&
$autoencoder/sequential/dense/BiasAdd�
$autoencoder/sequential/dense/SigmoidSigmoid-autoencoder/sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2&
$autoencoder/sequential/dense/Sigmoid�
8autoencoder/sequential/dense/ActivityRegularizer/SigmoidSigmoid(autoencoder/sequential/dense/Sigmoid:y:0*
T0*'
_output_shapes
:���������@2:
8autoencoder/sequential/dense/ActivityRegularizer/Sigmoid�
Gautoencoder/sequential/dense/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gautoencoder/sequential/dense/ActivityRegularizer/Mean/reduction_indices�
5autoencoder/sequential/dense/ActivityRegularizer/MeanMean<autoencoder/sequential/dense/ActivityRegularizer/Sigmoid:y:0Pautoencoder/sequential/dense/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
:@27
5autoencoder/sequential/dense/ActivityRegularizer/Mean�
:autoencoder/sequential/dense/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���.2<
:autoencoder/sequential/dense/ActivityRegularizer/Maximum/y�
8autoencoder/sequential/dense/ActivityRegularizer/MaximumMaximum>autoencoder/sequential/dense/ActivityRegularizer/Mean:output:0Cautoencoder/sequential/dense/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
:@2:
8autoencoder/sequential/dense/ActivityRegularizer/Maximum�
:autoencoder/sequential/dense/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2<
:autoencoder/sequential/dense/ActivityRegularizer/truediv/x�
8autoencoder/sequential/dense/ActivityRegularizer/truedivRealDivCautoencoder/sequential/dense/ActivityRegularizer/truediv/x:output:0<autoencoder/sequential/dense/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
:@2:
8autoencoder/sequential/dense/ActivityRegularizer/truediv�
4autoencoder/sequential/dense/ActivityRegularizer/LogLog<autoencoder/sequential/dense/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
:@26
4autoencoder/sequential/dense/ActivityRegularizer/Log�
6autoencoder/sequential/dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<28
6autoencoder/sequential/dense/ActivityRegularizer/mul/x�
4autoencoder/sequential/dense/ActivityRegularizer/mulMul?autoencoder/sequential/dense/ActivityRegularizer/mul/x:output:08autoencoder/sequential/dense/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
:@26
4autoencoder/sequential/dense/ActivityRegularizer/mul�
6autoencoder/sequential/dense/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?28
6autoencoder/sequential/dense/ActivityRegularizer/sub/x�
4autoencoder/sequential/dense/ActivityRegularizer/subSub?autoencoder/sequential/dense/ActivityRegularizer/sub/x:output:0<autoencoder/sequential/dense/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
:@26
4autoencoder/sequential/dense/ActivityRegularizer/sub�
<autoencoder/sequential/dense/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2>
<autoencoder/sequential/dense/ActivityRegularizer/truediv_1/x�
:autoencoder/sequential/dense/ActivityRegularizer/truediv_1RealDivEautoencoder/sequential/dense/ActivityRegularizer/truediv_1/x:output:08autoencoder/sequential/dense/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
:@2<
:autoencoder/sequential/dense/ActivityRegularizer/truediv_1�
6autoencoder/sequential/dense/ActivityRegularizer/Log_1Log>autoencoder/sequential/dense/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
:@28
6autoencoder/sequential/dense/ActivityRegularizer/Log_1�
8autoencoder/sequential/dense/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2:
8autoencoder/sequential/dense/ActivityRegularizer/mul_1/x�
6autoencoder/sequential/dense/ActivityRegularizer/mul_1MulAautoencoder/sequential/dense/ActivityRegularizer/mul_1/x:output:0:autoencoder/sequential/dense/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
:@28
6autoencoder/sequential/dense/ActivityRegularizer/mul_1�
4autoencoder/sequential/dense/ActivityRegularizer/addAddV28autoencoder/sequential/dense/ActivityRegularizer/mul:z:0:autoencoder/sequential/dense/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
:@26
4autoencoder/sequential/dense/ActivityRegularizer/add�
6autoencoder/sequential/dense/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 28
6autoencoder/sequential/dense/ActivityRegularizer/Const�
4autoencoder/sequential/dense/ActivityRegularizer/SumSum8autoencoder/sequential/dense/ActivityRegularizer/add:z:0?autoencoder/sequential/dense/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 26
4autoencoder/sequential/dense/ActivityRegularizer/Sum�
8autoencoder/sequential/dense/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2:
8autoencoder/sequential/dense/ActivityRegularizer/mul_2/x�
6autoencoder/sequential/dense/ActivityRegularizer/mul_2MulAautoencoder/sequential/dense/ActivityRegularizer/mul_2/x:output:0=autoencoder/sequential/dense/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 28
6autoencoder/sequential/dense/ActivityRegularizer/mul_2�
6autoencoder/sequential/dense/ActivityRegularizer/ShapeShape(autoencoder/sequential/dense/Sigmoid:y:0*
T0*
_output_shapes
:28
6autoencoder/sequential/dense/ActivityRegularizer/Shape�
Dautoencoder/sequential/dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dautoencoder/sequential/dense/ActivityRegularizer/strided_slice/stack�
Fautoencoder/sequential/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fautoencoder/sequential/dense/ActivityRegularizer/strided_slice/stack_1�
Fautoencoder/sequential/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fautoencoder/sequential/dense/ActivityRegularizer/strided_slice/stack_2�
>autoencoder/sequential/dense/ActivityRegularizer/strided_sliceStridedSlice?autoencoder/sequential/dense/ActivityRegularizer/Shape:output:0Mautoencoder/sequential/dense/ActivityRegularizer/strided_slice/stack:output:0Oautoencoder/sequential/dense/ActivityRegularizer/strided_slice/stack_1:output:0Oautoencoder/sequential/dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2@
>autoencoder/sequential/dense/ActivityRegularizer/strided_slice�
5autoencoder/sequential/dense/ActivityRegularizer/CastCastGautoencoder/sequential/dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 27
5autoencoder/sequential/dense/ActivityRegularizer/Cast�
:autoencoder/sequential/dense/ActivityRegularizer/truediv_2RealDiv:autoencoder/sequential/dense/ActivityRegularizer/mul_2:z:09autoencoder/sequential/dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2<
:autoencoder/sequential/dense/ActivityRegularizer/truediv_2�
6autoencoder/sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp?autoencoder_sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:@s*
dtype028
6autoencoder/sequential_1/dense_1/MatMul/ReadVariableOp�
'autoencoder/sequential_1/dense_1/MatMulMatMul(autoencoder/sequential/dense/Sigmoid:y:0>autoencoder/sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������s2)
'autoencoder/sequential_1/dense_1/MatMul�
7autoencoder/sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp@autoencoder_sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:s*
dtype029
7autoencoder/sequential_1/dense_1/BiasAdd/ReadVariableOp�
(autoencoder/sequential_1/dense_1/BiasAddBiasAdd1autoencoder/sequential_1/dense_1/MatMul:product:0?autoencoder/sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������s2*
(autoencoder/sequential_1/dense_1/BiasAdd�
(autoencoder/sequential_1/dense_1/SigmoidSigmoid1autoencoder/sequential_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������s2*
(autoencoder/sequential_1/dense_1/Sigmoid�
IdentityIdentity,autoencoder/sequential_1/dense_1/Sigmoid:y:04^autoencoder/sequential/dense/BiasAdd/ReadVariableOp3^autoencoder/sequential/dense/MatMul/ReadVariableOp8^autoencoder/sequential_1/dense_1/BiasAdd/ReadVariableOp7^autoencoder/sequential_1/dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������s2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������s: : : : 2j
3autoencoder/sequential/dense/BiasAdd/ReadVariableOp3autoencoder/sequential/dense/BiasAdd/ReadVariableOp2h
2autoencoder/sequential/dense/MatMul/ReadVariableOp2autoencoder/sequential/dense/MatMul/ReadVariableOp2r
7autoencoder/sequential_1/dense_1/BiasAdd/ReadVariableOp7autoencoder/sequential_1/dense_1/BiasAdd/ReadVariableOp2p
6autoencoder/sequential_1/dense_1/MatMul/ReadVariableOp6autoencoder/sequential_1/dense_1/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������s
!
_user_specified_name	input_1
�
�
+__inference_sequential_layer_call_fn_920479
input_1
unknown:s@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:���������@: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_9204612
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������s: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������s
!
_user_specified_name	input_1
�
�
H__inference_sequential_1_layer_call_and_return_conditional_losses_921177

inputs8
&dense_1_matmul_readvariableop_resource:@s5
'dense_1_biasadd_readvariableop_resource:s
identity��dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�0dense_1/kernel/Regularizer/Square/ReadVariableOp�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@s*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������s2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:s*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������s2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������s2
dense_1/Sigmoid�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@s*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@s2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
IdentityIdentitydense_1/Sigmoid:y:0^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������s2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�?
�
F__inference_sequential_layer_call_and_return_conditional_losses_921118

inputs6
$dense_matmul_readvariableop_resource:s@3
%dense_biasadd_readvariableop_resource:@
identity

identity_1��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:s@*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense/BiasAdds
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
dense/Sigmoid�
!dense/ActivityRegularizer/SigmoidSigmoiddense/Sigmoid:y:0*
T0*'
_output_shapes
:���������@2#
!dense/ActivityRegularizer/Sigmoid�
0dense/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 22
0dense/ActivityRegularizer/Mean/reduction_indices�
dense/ActivityRegularizer/MeanMean%dense/ActivityRegularizer/Sigmoid:y:09dense/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
:@2 
dense/ActivityRegularizer/Mean�
#dense/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���.2%
#dense/ActivityRegularizer/Maximum/y�
!dense/ActivityRegularizer/MaximumMaximum'dense/ActivityRegularizer/Mean:output:0,dense/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
:@2#
!dense/ActivityRegularizer/Maximum�
#dense/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2%
#dense/ActivityRegularizer/truediv/x�
!dense/ActivityRegularizer/truedivRealDiv,dense/ActivityRegularizer/truediv/x:output:0%dense/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
:@2#
!dense/ActivityRegularizer/truediv�
dense/ActivityRegularizer/LogLog%dense/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
:@2
dense/ActivityRegularizer/Log�
dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2!
dense/ActivityRegularizer/mul/x�
dense/ActivityRegularizer/mulMul(dense/ActivityRegularizer/mul/x:output:0!dense/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
:@2
dense/ActivityRegularizer/mul�
dense/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2!
dense/ActivityRegularizer/sub/x�
dense/ActivityRegularizer/subSub(dense/ActivityRegularizer/sub/x:output:0%dense/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
:@2
dense/ActivityRegularizer/sub�
%dense/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2'
%dense/ActivityRegularizer/truediv_1/x�
#dense/ActivityRegularizer/truediv_1RealDiv.dense/ActivityRegularizer/truediv_1/x:output:0!dense/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
:@2%
#dense/ActivityRegularizer/truediv_1�
dense/ActivityRegularizer/Log_1Log'dense/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
:@2!
dense/ActivityRegularizer/Log_1�
!dense/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2#
!dense/ActivityRegularizer/mul_1/x�
dense/ActivityRegularizer/mul_1Mul*dense/ActivityRegularizer/mul_1/x:output:0#dense/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
:@2!
dense/ActivityRegularizer/mul_1�
dense/ActivityRegularizer/addAddV2!dense/ActivityRegularizer/mul:z:0#dense/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
:@2
dense/ActivityRegularizer/add�
dense/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense/ActivityRegularizer/Const�
dense/ActivityRegularizer/SumSum!dense/ActivityRegularizer/add:z:0(dense/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/ActivityRegularizer/Sum�
!dense/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2#
!dense/ActivityRegularizer/mul_2/x�
dense/ActivityRegularizer/mul_2Mul*dense/ActivityRegularizer/mul_2/x:output:0&dense/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense/ActivityRegularizer/mul_2�
dense/ActivityRegularizer/ShapeShapedense/Sigmoid:y:0*
T0*
_output_shapes
:2!
dense/ActivityRegularizer/Shape�
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense/ActivityRegularizer/strided_slice/stack�
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_1�
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_2�
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense/ActivityRegularizer/strided_slice�
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
dense/ActivityRegularizer/Cast�
#dense/ActivityRegularizer/truediv_2RealDiv#dense/ActivityRegularizer/mul_2:z:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense/ActivityRegularizer/truediv_2�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:s@*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:s@2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
IdentityIdentitydense/Sigmoid:y:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity'dense/ActivityRegularizer/truediv_2:z:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������s: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������s
 
_user_specified_nameinputs
�
�
H__inference_sequential_1_layer_call_and_return_conditional_losses_920564

inputs 
dense_1_920552:@s
dense_1_920554:s
identity��dense_1/StatefulPartitionedCall�0dense_1/kernel/Regularizer/Square/ReadVariableOp�
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_920552dense_1_920554*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������s*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_9205512!
dense_1/StatefulPartitionedCall�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_920552*
_output_shapes

:@s*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@s2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������s2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
,__inference_autoencoder_layer_call_fn_920878
x
unknown:s@
	unknown_0:@
	unknown_1:@s
	unknown_2:s
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:���������s: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_autoencoder_layer_call_and_return_conditional_losses_9207412
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������s2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������s: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:���������s

_user_specified_nameX
�
�
__inference_loss_fn_1_921308K
9dense_1_kernel_regularizer_square_readvariableop_resource:@s
identity��0dense_1/kernel/Regularizer/Square/ReadVariableOp�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_1_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:@s*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@s2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
IdentityIdentity"dense_1/kernel/Regularizer/mul:z:01^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp
�
�
E__inference_dense_layer_call_and_return_all_conditional_losses_921254

inputs
unknown:s@
	unknown_0:@
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_9203732
StatefulPartitionedCall�
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
GPU 2J 8� *6
f1R/
-__inference_dense_activity_regularizer_9203492
PartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

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
:���������s: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������s
 
_user_specified_nameinputs
�`
�
G__inference_autoencoder_layer_call_and_return_conditional_losses_920938
xA
/sequential_dense_matmul_readvariableop_resource:s@>
0sequential_dense_biasadd_readvariableop_resource:@E
3sequential_1_dense_1_matmul_readvariableop_resource:@sB
4sequential_1_dense_1_biasadd_readvariableop_resource:s
identity

identity_1��.dense/kernel/Regularizer/Square/ReadVariableOp�0dense_1/kernel/Regularizer/Square/ReadVariableOp�'sequential/dense/BiasAdd/ReadVariableOp�&sequential/dense/MatMul/ReadVariableOp�+sequential_1/dense_1/BiasAdd/ReadVariableOp�*sequential_1/dense_1/MatMul/ReadVariableOp�
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:s@*
dtype02(
&sequential/dense/MatMul/ReadVariableOp�
sequential/dense/MatMulMatMulx.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
sequential/dense/MatMul�
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp�
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
sequential/dense/BiasAdd�
sequential/dense/SigmoidSigmoid!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
sequential/dense/Sigmoid�
,sequential/dense/ActivityRegularizer/SigmoidSigmoidsequential/dense/Sigmoid:y:0*
T0*'
_output_shapes
:���������@2.
,sequential/dense/ActivityRegularizer/Sigmoid�
;sequential/dense/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2=
;sequential/dense/ActivityRegularizer/Mean/reduction_indices�
)sequential/dense/ActivityRegularizer/MeanMean0sequential/dense/ActivityRegularizer/Sigmoid:y:0Dsequential/dense/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
:@2+
)sequential/dense/ActivityRegularizer/Mean�
.sequential/dense/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���.20
.sequential/dense/ActivityRegularizer/Maximum/y�
,sequential/dense/ActivityRegularizer/MaximumMaximum2sequential/dense/ActivityRegularizer/Mean:output:07sequential/dense/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
:@2.
,sequential/dense/ActivityRegularizer/Maximum�
.sequential/dense/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.sequential/dense/ActivityRegularizer/truediv/x�
,sequential/dense/ActivityRegularizer/truedivRealDiv7sequential/dense/ActivityRegularizer/truediv/x:output:00sequential/dense/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
:@2.
,sequential/dense/ActivityRegularizer/truediv�
(sequential/dense/ActivityRegularizer/LogLog0sequential/dense/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
:@2*
(sequential/dense/ActivityRegularizer/Log�
*sequential/dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2,
*sequential/dense/ActivityRegularizer/mul/x�
(sequential/dense/ActivityRegularizer/mulMul3sequential/dense/ActivityRegularizer/mul/x:output:0,sequential/dense/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
:@2*
(sequential/dense/ActivityRegularizer/mul�
*sequential/dense/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2,
*sequential/dense/ActivityRegularizer/sub/x�
(sequential/dense/ActivityRegularizer/subSub3sequential/dense/ActivityRegularizer/sub/x:output:00sequential/dense/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
:@2*
(sequential/dense/ActivityRegularizer/sub�
0sequential/dense/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?22
0sequential/dense/ActivityRegularizer/truediv_1/x�
.sequential/dense/ActivityRegularizer/truediv_1RealDiv9sequential/dense/ActivityRegularizer/truediv_1/x:output:0,sequential/dense/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
:@20
.sequential/dense/ActivityRegularizer/truediv_1�
*sequential/dense/ActivityRegularizer/Log_1Log2sequential/dense/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
:@2,
*sequential/dense/ActivityRegularizer/Log_1�
,sequential/dense/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2.
,sequential/dense/ActivityRegularizer/mul_1/x�
*sequential/dense/ActivityRegularizer/mul_1Mul5sequential/dense/ActivityRegularizer/mul_1/x:output:0.sequential/dense/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
:@2,
*sequential/dense/ActivityRegularizer/mul_1�
(sequential/dense/ActivityRegularizer/addAddV2,sequential/dense/ActivityRegularizer/mul:z:0.sequential/dense/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
:@2*
(sequential/dense/ActivityRegularizer/add�
*sequential/dense/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential/dense/ActivityRegularizer/Const�
(sequential/dense/ActivityRegularizer/SumSum,sequential/dense/ActivityRegularizer/add:z:03sequential/dense/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2*
(sequential/dense/ActivityRegularizer/Sum�
,sequential/dense/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2.
,sequential/dense/ActivityRegularizer/mul_2/x�
*sequential/dense/ActivityRegularizer/mul_2Mul5sequential/dense/ActivityRegularizer/mul_2/x:output:01sequential/dense/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*sequential/dense/ActivityRegularizer/mul_2�
*sequential/dense/ActivityRegularizer/ShapeShapesequential/dense/Sigmoid:y:0*
T0*
_output_shapes
:2,
*sequential/dense/ActivityRegularizer/Shape�
8sequential/dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8sequential/dense/ActivityRegularizer/strided_slice/stack�
:sequential/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:sequential/dense/ActivityRegularizer/strided_slice/stack_1�
:sequential/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:sequential/dense/ActivityRegularizer/strided_slice/stack_2�
2sequential/dense/ActivityRegularizer/strided_sliceStridedSlice3sequential/dense/ActivityRegularizer/Shape:output:0Asequential/dense/ActivityRegularizer/strided_slice/stack:output:0Csequential/dense/ActivityRegularizer/strided_slice/stack_1:output:0Csequential/dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2sequential/dense/ActivityRegularizer/strided_slice�
)sequential/dense/ActivityRegularizer/CastCast;sequential/dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2+
)sequential/dense/ActivityRegularizer/Cast�
.sequential/dense/ActivityRegularizer/truediv_2RealDiv.sequential/dense/ActivityRegularizer/mul_2:z:0-sequential/dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 20
.sequential/dense/ActivityRegularizer/truediv_2�
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:@s*
dtype02,
*sequential_1/dense_1/MatMul/ReadVariableOp�
sequential_1/dense_1/MatMulMatMulsequential/dense/Sigmoid:y:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������s2
sequential_1/dense_1/MatMul�
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:s*
dtype02-
+sequential_1/dense_1/BiasAdd/ReadVariableOp�
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������s2
sequential_1/dense_1/BiasAdd�
sequential_1/dense_1/SigmoidSigmoid%sequential_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������s2
sequential_1/dense_1/Sigmoid�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:s@*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:s@2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:@s*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@s2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
IdentityIdentity sequential_1/dense_1/Sigmoid:y:0/^dense/kernel/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������s2

Identity�

Identity_1Identity2sequential/dense/ActivityRegularizer/truediv_2:z:0/^dense/kernel/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������s: : : : 2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp:J F
'
_output_shapes
:���������s

_user_specified_nameX
�
�
,__inference_autoencoder_layer_call_fn_920767
input_1
unknown:s@
	unknown_0:@
	unknown_1:@s
	unknown_2:s
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:���������s: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_autoencoder_layer_call_and_return_conditional_losses_9207412
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������s2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������s: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������s
!
_user_specified_name	input_1
�#
�
G__inference_autoencoder_layer_call_and_return_conditional_losses_920795
input_1#
sequential_920770:s@
sequential_920772:@%
sequential_1_920776:@s!
sequential_1_920778:s
identity

identity_1��.dense/kernel/Regularizer/Square/ReadVariableOp�0dense_1/kernel/Regularizer/Square/ReadVariableOp�"sequential/StatefulPartitionedCall�$sequential_1/StatefulPartitionedCall�
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_920770sequential_920772*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:���������@: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_9203952$
"sequential/StatefulPartitionedCall�
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_920776sequential_1_920778*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������s*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_9205642&
$sequential_1/StatefulPartitionedCall�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_920770*
_output_shapes

:s@*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:s@2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_1_920776*
_output_shapes

:@s*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@s2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0/^dense/kernel/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������s2

Identity�

Identity_1Identity+sequential/StatefulPartitionedCall:output:1/^dense/kernel/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������s: : : : 2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:P L
'
_output_shapes
:���������s
!
_user_specified_name	input_1
�
�
__inference__traced_save_921360
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
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
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
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
$: :s@:@:@s:s: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:s@: 

_output_shapes
:@:$ 

_output_shapes

:@s: 

_output_shapes
:s:

_output_shapes
: 
�
�
-__inference_sequential_1_layer_call_fn_921142

inputs
unknown:@s
	unknown_0:s
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������s*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_9205642
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������s2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
A__inference_dense_layer_call_and_return_conditional_losses_920373

inputs0
matmul_readvariableop_resource:s@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�.dense/kernel/Regularizer/Square/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:s@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������@2	
Sigmoid�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:s@*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:s@2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������s: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������s
 
_user_specified_nameinputs
�
�
+__inference_sequential_layer_call_fn_920403
input_1
unknown:s@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:���������@: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_9203952
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������s: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������s
!
_user_specified_name	input_1
�!
�
F__inference_sequential_layer_call_and_return_conditional_losses_920461

inputs
dense_920440:s@
dense_920442:@
identity

identity_1��dense/StatefulPartitionedCall�.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_920440dense_920442*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_9203732
dense/StatefulPartitionedCall�
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *6
f1R/
-__inference_dense_activity_regularizer_9203492+
)dense/ActivityRegularizer/PartitionedCall�
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2!
dense/ActivityRegularizer/Shape�
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense/ActivityRegularizer/strided_slice/stack�
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_1�
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_2�
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense/ActivityRegularizer/strided_slice�
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
dense/ActivityRegularizer/Cast�
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2#
!dense/ActivityRegularizer/truediv�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_920440*
_output_shapes

:s@*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:s@2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������s: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������s
 
_user_specified_nameinputs
�
�
-__inference_sequential_1_layer_call_fn_921133
dense_1_input
unknown:@s
	unknown_0:s
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_1_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������s*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_9205642
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������s2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:���������@
'
_user_specified_namedense_1_input
�#
�
G__inference_autoencoder_layer_call_and_return_conditional_losses_920823
input_1#
sequential_920798:s@
sequential_920800:@%
sequential_1_920804:@s!
sequential_1_920806:s
identity

identity_1��.dense/kernel/Regularizer/Square/ReadVariableOp�0dense_1/kernel/Regularizer/Square/ReadVariableOp�"sequential/StatefulPartitionedCall�$sequential_1/StatefulPartitionedCall�
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_920798sequential_920800*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:���������@: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_9204612$
"sequential/StatefulPartitionedCall�
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_920804sequential_1_920806*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������s*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_9206072&
$sequential_1/StatefulPartitionedCall�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_920798*
_output_shapes

:s@*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:s@2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_1_920804*
_output_shapes

:@s*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@s2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0/^dense/kernel/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������s2

Identity�

Identity_1Identity+sequential/StatefulPartitionedCall:output:1/^dense/kernel/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������s: : : : 2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:P L
'
_output_shapes
:���������s
!
_user_specified_name	input_1
�?
�
F__inference_sequential_layer_call_and_return_conditional_losses_921071

inputs6
$dense_matmul_readvariableop_resource:s@3
%dense_biasadd_readvariableop_resource:@
identity

identity_1��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:s@*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense/BiasAdds
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
dense/Sigmoid�
!dense/ActivityRegularizer/SigmoidSigmoiddense/Sigmoid:y:0*
T0*'
_output_shapes
:���������@2#
!dense/ActivityRegularizer/Sigmoid�
0dense/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 22
0dense/ActivityRegularizer/Mean/reduction_indices�
dense/ActivityRegularizer/MeanMean%dense/ActivityRegularizer/Sigmoid:y:09dense/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
:@2 
dense/ActivityRegularizer/Mean�
#dense/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���.2%
#dense/ActivityRegularizer/Maximum/y�
!dense/ActivityRegularizer/MaximumMaximum'dense/ActivityRegularizer/Mean:output:0,dense/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
:@2#
!dense/ActivityRegularizer/Maximum�
#dense/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2%
#dense/ActivityRegularizer/truediv/x�
!dense/ActivityRegularizer/truedivRealDiv,dense/ActivityRegularizer/truediv/x:output:0%dense/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
:@2#
!dense/ActivityRegularizer/truediv�
dense/ActivityRegularizer/LogLog%dense/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
:@2
dense/ActivityRegularizer/Log�
dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2!
dense/ActivityRegularizer/mul/x�
dense/ActivityRegularizer/mulMul(dense/ActivityRegularizer/mul/x:output:0!dense/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
:@2
dense/ActivityRegularizer/mul�
dense/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2!
dense/ActivityRegularizer/sub/x�
dense/ActivityRegularizer/subSub(dense/ActivityRegularizer/sub/x:output:0%dense/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
:@2
dense/ActivityRegularizer/sub�
%dense/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2'
%dense/ActivityRegularizer/truediv_1/x�
#dense/ActivityRegularizer/truediv_1RealDiv.dense/ActivityRegularizer/truediv_1/x:output:0!dense/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
:@2%
#dense/ActivityRegularizer/truediv_1�
dense/ActivityRegularizer/Log_1Log'dense/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
:@2!
dense/ActivityRegularizer/Log_1�
!dense/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2#
!dense/ActivityRegularizer/mul_1/x�
dense/ActivityRegularizer/mul_1Mul*dense/ActivityRegularizer/mul_1/x:output:0#dense/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
:@2!
dense/ActivityRegularizer/mul_1�
dense/ActivityRegularizer/addAddV2!dense/ActivityRegularizer/mul:z:0#dense/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
:@2
dense/ActivityRegularizer/add�
dense/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense/ActivityRegularizer/Const�
dense/ActivityRegularizer/SumSum!dense/ActivityRegularizer/add:z:0(dense/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/ActivityRegularizer/Sum�
!dense/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2#
!dense/ActivityRegularizer/mul_2/x�
dense/ActivityRegularizer/mul_2Mul*dense/ActivityRegularizer/mul_2/x:output:0&dense/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense/ActivityRegularizer/mul_2�
dense/ActivityRegularizer/ShapeShapedense/Sigmoid:y:0*
T0*
_output_shapes
:2!
dense/ActivityRegularizer/Shape�
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense/ActivityRegularizer/strided_slice/stack�
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_1�
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_2�
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense/ActivityRegularizer/strided_slice�
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
dense/ActivityRegularizer/Cast�
#dense/ActivityRegularizer/truediv_2RealDiv#dense/ActivityRegularizer/mul_2:z:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense/ActivityRegularizer/truediv_2�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:s@*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:s@2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
IdentityIdentitydense/Sigmoid:y:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity'dense/ActivityRegularizer/truediv_2:z:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������s: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������s
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_920850
input_1
unknown:s@
	unknown_0:@
	unknown_1:@s
	unknown_2:s
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������s*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_9203192
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������s2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������s: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������s
!
_user_specified_name	input_1
�#
�
G__inference_autoencoder_layer_call_and_return_conditional_losses_920741
x#
sequential_920716:s@
sequential_920718:@%
sequential_1_920722:@s!
sequential_1_920724:s
identity

identity_1��.dense/kernel/Regularizer/Square/ReadVariableOp�0dense_1/kernel/Regularizer/Square/ReadVariableOp�"sequential/StatefulPartitionedCall�$sequential_1/StatefulPartitionedCall�
"sequential/StatefulPartitionedCallStatefulPartitionedCallxsequential_920716sequential_920718*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:���������@: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_9204612$
"sequential/StatefulPartitionedCall�
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_920722sequential_1_920724*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������s*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_9206072&
$sequential_1/StatefulPartitionedCall�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_920716*
_output_shapes

:s@*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:s@2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_1_920722*
_output_shapes

:@s*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@s2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0/^dense/kernel/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������s2

Identity�

Identity_1Identity+sequential/StatefulPartitionedCall:output:1/^dense/kernel/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������s: : : : 2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:J F
'
_output_shapes
:���������s

_user_specified_nameX
�
�
H__inference_sequential_1_layer_call_and_return_conditional_losses_921228
dense_1_input8
&dense_1_matmul_readvariableop_resource:@s5
'dense_1_biasadd_readvariableop_resource:s
identity��dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�0dense_1/kernel/Regularizer/Square/ReadVariableOp�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@s*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldense_1_input%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������s2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:s*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������s2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������s2
dense_1/Sigmoid�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@s*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@s2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
IdentityIdentitydense_1/Sigmoid:y:0^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������s2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:V R
'
_output_shapes
:���������@
'
_user_specified_namedense_1_input
�!
�
F__inference_sequential_layer_call_and_return_conditional_losses_920527
input_1
dense_920506:s@
dense_920508:@
identity

identity_1��dense/StatefulPartitionedCall�.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_920506dense_920508*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_9203732
dense/StatefulPartitionedCall�
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *6
f1R/
-__inference_dense_activity_regularizer_9203492+
)dense/ActivityRegularizer/PartitionedCall�
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2!
dense/ActivityRegularizer/Shape�
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense/ActivityRegularizer/strided_slice/stack�
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_1�
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_2�
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense/ActivityRegularizer/strided_slice�
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
dense/ActivityRegularizer/Cast�
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2#
!dense/ActivityRegularizer/truediv�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_920506*
_output_shapes

:s@*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:s@2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������s: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:���������s
!
_user_specified_name	input_1
�
�
+__inference_sequential_layer_call_fn_921024

inputs
unknown:s@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:���������@: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_9204612
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������s: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������s
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_921265I
7dense_kernel_regularizer_square_readvariableop_resource:s@
identity��.dense/kernel/Regularizer/Square/ReadVariableOp�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7dense_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:s@*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:s@2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
IdentityIdentity dense/kernel/Regularizer/mul:z:0/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp
�`
�
G__inference_autoencoder_layer_call_and_return_conditional_losses_920998
xA
/sequential_dense_matmul_readvariableop_resource:s@>
0sequential_dense_biasadd_readvariableop_resource:@E
3sequential_1_dense_1_matmul_readvariableop_resource:@sB
4sequential_1_dense_1_biasadd_readvariableop_resource:s
identity

identity_1��.dense/kernel/Regularizer/Square/ReadVariableOp�0dense_1/kernel/Regularizer/Square/ReadVariableOp�'sequential/dense/BiasAdd/ReadVariableOp�&sequential/dense/MatMul/ReadVariableOp�+sequential_1/dense_1/BiasAdd/ReadVariableOp�*sequential_1/dense_1/MatMul/ReadVariableOp�
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:s@*
dtype02(
&sequential/dense/MatMul/ReadVariableOp�
sequential/dense/MatMulMatMulx.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
sequential/dense/MatMul�
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp�
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
sequential/dense/BiasAdd�
sequential/dense/SigmoidSigmoid!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
sequential/dense/Sigmoid�
,sequential/dense/ActivityRegularizer/SigmoidSigmoidsequential/dense/Sigmoid:y:0*
T0*'
_output_shapes
:���������@2.
,sequential/dense/ActivityRegularizer/Sigmoid�
;sequential/dense/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2=
;sequential/dense/ActivityRegularizer/Mean/reduction_indices�
)sequential/dense/ActivityRegularizer/MeanMean0sequential/dense/ActivityRegularizer/Sigmoid:y:0Dsequential/dense/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
:@2+
)sequential/dense/ActivityRegularizer/Mean�
.sequential/dense/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���.20
.sequential/dense/ActivityRegularizer/Maximum/y�
,sequential/dense/ActivityRegularizer/MaximumMaximum2sequential/dense/ActivityRegularizer/Mean:output:07sequential/dense/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
:@2.
,sequential/dense/ActivityRegularizer/Maximum�
.sequential/dense/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<20
.sequential/dense/ActivityRegularizer/truediv/x�
,sequential/dense/ActivityRegularizer/truedivRealDiv7sequential/dense/ActivityRegularizer/truediv/x:output:00sequential/dense/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
:@2.
,sequential/dense/ActivityRegularizer/truediv�
(sequential/dense/ActivityRegularizer/LogLog0sequential/dense/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
:@2*
(sequential/dense/ActivityRegularizer/Log�
*sequential/dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2,
*sequential/dense/ActivityRegularizer/mul/x�
(sequential/dense/ActivityRegularizer/mulMul3sequential/dense/ActivityRegularizer/mul/x:output:0,sequential/dense/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
:@2*
(sequential/dense/ActivityRegularizer/mul�
*sequential/dense/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2,
*sequential/dense/ActivityRegularizer/sub/x�
(sequential/dense/ActivityRegularizer/subSub3sequential/dense/ActivityRegularizer/sub/x:output:00sequential/dense/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
:@2*
(sequential/dense/ActivityRegularizer/sub�
0sequential/dense/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?22
0sequential/dense/ActivityRegularizer/truediv_1/x�
.sequential/dense/ActivityRegularizer/truediv_1RealDiv9sequential/dense/ActivityRegularizer/truediv_1/x:output:0,sequential/dense/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
:@20
.sequential/dense/ActivityRegularizer/truediv_1�
*sequential/dense/ActivityRegularizer/Log_1Log2sequential/dense/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
:@2,
*sequential/dense/ActivityRegularizer/Log_1�
,sequential/dense/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2.
,sequential/dense/ActivityRegularizer/mul_1/x�
*sequential/dense/ActivityRegularizer/mul_1Mul5sequential/dense/ActivityRegularizer/mul_1/x:output:0.sequential/dense/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
:@2,
*sequential/dense/ActivityRegularizer/mul_1�
(sequential/dense/ActivityRegularizer/addAddV2,sequential/dense/ActivityRegularizer/mul:z:0.sequential/dense/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
:@2*
(sequential/dense/ActivityRegularizer/add�
*sequential/dense/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential/dense/ActivityRegularizer/Const�
(sequential/dense/ActivityRegularizer/SumSum,sequential/dense/ActivityRegularizer/add:z:03sequential/dense/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2*
(sequential/dense/ActivityRegularizer/Sum�
,sequential/dense/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2.
,sequential/dense/ActivityRegularizer/mul_2/x�
*sequential/dense/ActivityRegularizer/mul_2Mul5sequential/dense/ActivityRegularizer/mul_2/x:output:01sequential/dense/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*sequential/dense/ActivityRegularizer/mul_2�
*sequential/dense/ActivityRegularizer/ShapeShapesequential/dense/Sigmoid:y:0*
T0*
_output_shapes
:2,
*sequential/dense/ActivityRegularizer/Shape�
8sequential/dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8sequential/dense/ActivityRegularizer/strided_slice/stack�
:sequential/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:sequential/dense/ActivityRegularizer/strided_slice/stack_1�
:sequential/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:sequential/dense/ActivityRegularizer/strided_slice/stack_2�
2sequential/dense/ActivityRegularizer/strided_sliceStridedSlice3sequential/dense/ActivityRegularizer/Shape:output:0Asequential/dense/ActivityRegularizer/strided_slice/stack:output:0Csequential/dense/ActivityRegularizer/strided_slice/stack_1:output:0Csequential/dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2sequential/dense/ActivityRegularizer/strided_slice�
)sequential/dense/ActivityRegularizer/CastCast;sequential/dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2+
)sequential/dense/ActivityRegularizer/Cast�
.sequential/dense/ActivityRegularizer/truediv_2RealDiv.sequential/dense/ActivityRegularizer/mul_2:z:0-sequential/dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 20
.sequential/dense/ActivityRegularizer/truediv_2�
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:@s*
dtype02,
*sequential_1/dense_1/MatMul/ReadVariableOp�
sequential_1/dense_1/MatMulMatMulsequential/dense/Sigmoid:y:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������s2
sequential_1/dense_1/MatMul�
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:s*
dtype02-
+sequential_1/dense_1/BiasAdd/ReadVariableOp�
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������s2
sequential_1/dense_1/BiasAdd�
sequential_1/dense_1/SigmoidSigmoid%sequential_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������s2
sequential_1/dense_1/Sigmoid�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:s@*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:s@2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:@s*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@s2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
IdentityIdentity sequential_1/dense_1/Sigmoid:y:0/^dense/kernel/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������s2

Identity�

Identity_1Identity2sequential/dense/ActivityRegularizer/truediv_2:z:0/^dense/kernel/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������s: : : : 2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp:J F
'
_output_shapes
:���������s

_user_specified_nameX
�
�
C__inference_dense_1_layer_call_and_return_conditional_losses_921297

inputs0
matmul_readvariableop_resource:@s-
biasadd_readvariableop_resource:s
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�0dense_1/kernel/Regularizer/Square/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@s*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������s2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:s*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������s2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������s2	
Sigmoid�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@s*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@s2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������s2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
,__inference_autoencoder_layer_call_fn_920697
input_1
unknown:s@
	unknown_0:@
	unknown_1:@s
	unknown_2:s
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:���������s: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_autoencoder_layer_call_and_return_conditional_losses_9206852
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������s2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������s: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������s
!
_user_specified_name	input_1
�
M
-__inference_dense_activity_regularizer_920349

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
 *���.2
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
�#<2
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
�#<2
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
 *  �?2
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
 *�p}?2
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
 *�p}?2	
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
:���������2
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
 *  �?2	
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
�
�
H__inference_sequential_1_layer_call_and_return_conditional_losses_921194

inputs8
&dense_1_matmul_readvariableop_resource:@s5
'dense_1_biasadd_readvariableop_resource:s
identity��dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�0dense_1/kernel/Regularizer/Square/ReadVariableOp�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@s*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������s2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:s*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������s2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������s2
dense_1/Sigmoid�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@s*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@s2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
IdentityIdentitydense_1/Sigmoid:y:0^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������s2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
A__inference_dense_layer_call_and_return_conditional_losses_921325

inputs0
matmul_readvariableop_resource:s@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�.dense/kernel/Regularizer/Square/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:s@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������@2	
Sigmoid�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:s@*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:s@2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������s: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������s
 
_user_specified_nameinputs
�
�
+__inference_sequential_layer_call_fn_921014

inputs
unknown:s@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:���������@: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_9203952
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������s: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������s
 
_user_specified_nameinputs
�#
�
G__inference_autoencoder_layer_call_and_return_conditional_losses_920685
x#
sequential_920660:s@
sequential_920662:@%
sequential_1_920666:@s!
sequential_1_920668:s
identity

identity_1��.dense/kernel/Regularizer/Square/ReadVariableOp�0dense_1/kernel/Regularizer/Square/ReadVariableOp�"sequential/StatefulPartitionedCall�$sequential_1/StatefulPartitionedCall�
"sequential/StatefulPartitionedCallStatefulPartitionedCallxsequential_920660sequential_920662*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:���������@: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_9203952$
"sequential/StatefulPartitionedCall�
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_920666sequential_1_920668*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������s*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_9205642&
$sequential_1/StatefulPartitionedCall�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_920660*
_output_shapes

:s@*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:s@2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_1_920666*
_output_shapes

:@s*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@s2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0/^dense/kernel/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������s2

Identity�

Identity_1Identity+sequential/StatefulPartitionedCall:output:1/^dense/kernel/Regularizer/Square/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������s: : : : 2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:J F
'
_output_shapes
:���������s

_user_specified_nameX
�
�
C__inference_dense_1_layer_call_and_return_conditional_losses_920551

inputs0
matmul_readvariableop_resource:@s-
biasadd_readvariableop_resource:s
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�0dense_1/kernel/Regularizer/Square/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@s*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������s2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:s*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������s2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������s2	
Sigmoid�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@s*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@s2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������s2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
,__inference_autoencoder_layer_call_fn_920864
x
unknown:s@
	unknown_0:@
	unknown_1:@s
	unknown_2:s
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:���������s: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_autoencoder_layer_call_and_return_conditional_losses_9206852
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������s2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������s: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:���������s

_user_specified_nameX
�
�
-__inference_sequential_1_layer_call_fn_921151

inputs
unknown:@s
	unknown_0:s
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������s*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_9206072
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������s2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
H__inference_sequential_1_layer_call_and_return_conditional_losses_921211
dense_1_input8
&dense_1_matmul_readvariableop_resource:@s5
'dense_1_biasadd_readvariableop_resource:s
identity��dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�0dense_1/kernel/Regularizer/Square/ReadVariableOp�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@s*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldense_1_input%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������s2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:s*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������s2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������s2
dense_1/Sigmoid�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@s*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@s2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
IdentityIdentitydense_1/Sigmoid:y:0^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������s2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:V R
'
_output_shapes
:���������@
'
_user_specified_namedense_1_input
�
�
(__inference_dense_1_layer_call_fn_921280

inputs
unknown:@s
	unknown_0:s
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������s*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_9205512
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������s2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
H__inference_sequential_1_layer_call_and_return_conditional_losses_920607

inputs 
dense_1_920595:@s
dense_1_920597:s
identity��dense_1/StatefulPartitionedCall�0dense_1/kernel/Regularizer/Square/ReadVariableOp�
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_920595dense_1_920597*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������s*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_9205512!
dense_1/StatefulPartitionedCall�
0dense_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_1_920595*
_output_shapes

:@s*
dtype022
0dense_1/kernel/Regularizer/Square/ReadVariableOp�
!dense_1/kernel/Regularizer/SquareSquare8dense_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@s2#
!dense_1/kernel/Regularizer/Square�
 dense_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_1/kernel/Regularizer/Const�
dense_1/kernel/Regularizer/SumSum%dense_1/kernel/Regularizer/Square:y:0)dense_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/Sum�
 dense_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 dense_1/kernel/Regularizer/mul/x�
dense_1/kernel/Regularizer/mulMul)dense_1/kernel/Regularizer/mul/x:output:0'dense_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_1/kernel/Regularizer/mul�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall1^dense_1/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������s2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2d
0dense_1/kernel/Regularizer/Square/ReadVariableOp0dense_1/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�!
�
F__inference_sequential_layer_call_and_return_conditional_losses_920395

inputs
dense_920374:s@
dense_920376:@
identity

identity_1��dense/StatefulPartitionedCall�.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_920374dense_920376*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_9203732
dense/StatefulPartitionedCall�
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *6
f1R/
-__inference_dense_activity_regularizer_9203492+
)dense/ActivityRegularizer/PartitionedCall�
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2!
dense/ActivityRegularizer/Shape�
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense/ActivityRegularizer/strided_slice/stack�
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_1�
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_2�
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense/ActivityRegularizer/strided_slice�
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
dense/ActivityRegularizer/Cast�
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2#
!dense/ActivityRegularizer/truediv�
.dense/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_920374*
_output_shapes

:s@*
dtype020
.dense/kernel/Regularizer/Square/ReadVariableOp�
dense/kernel/Regularizer/SquareSquare6dense/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:s@2!
dense/kernel/Regularizer/Square�
dense/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
dense/kernel/Regularizer/Const�
dense/kernel/Regularizer/SumSum#dense/kernel/Regularizer/Square:y:0'dense/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/Sum�
dense/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
dense/kernel/Regularizer/mul/x�
dense/kernel/Regularizer/mulMul'dense/kernel/Regularizer/mul/x:output:0%dense/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/kernel/Regularizer/mul�
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity�

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall/^dense/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������s: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2`
.dense/kernel/Regularizer/Square/ReadVariableOp.dense/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������s
 
_user_specified_nameinputs
�
�
"__inference__traced_restore_921382
file_prefix/
assignvariableop_dense_kernel:s@+
assignvariableop_1_dense_bias:@3
!assignvariableop_2_dense_1_kernel:@s-
assignvariableop_3_dense_1_bias:s

identity_5��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slices�
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

Identity�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4�

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
_user_specified_namefile_prefix"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������s<
output_10
StatefulPartitionedCall:0���������stensorflow/serving/predict:��
�
history
encoder
decoder
trainable_variables
regularization_losses
	variables
	keras_api

signatures
8_default_save_signature
9__call__
*:&call_and_return_all_conditional_losses"�
_tf_keras_model�{"name": "autoencoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Autoencoder", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 115]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Autoencoder"}}
 "
trackable_dict_wrapper
�
	layer_with_weights-0
	layer-0

trainable_variables
regularization_losses
	variables
	keras_api
;__call__
*<&call_and_return_all_conditional_losses"�
_tf_keras_sequential�{"name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 115]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}}, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 115}}, "shared_object_id": 6}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 115]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 115]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 115]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}]}}}
�
layer_with_weights-0
layer-0
trainable_variables
regularization_losses
	variables
	keras_api
=__call__
*>&call_and_return_all_conditional_losses"�
_tf_keras_sequential�{"name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_1_input"}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 115, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [115, 64]}, "float32", "dense_1_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_1_input"}, "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 115, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8}]}}}
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
�
trainable_variables
layer_metrics
non_trainable_variables
layer_regularization_losses
regularization_losses
metrics
	variables

layers
9__call__
8_default_save_signature
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
,
?serving_default"
signature_map
�

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
@__call__
*A&call_and_return_all_conditional_losses"�

_tf_keras_layer�	{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 115}}, "shared_object_id": 6}, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 115]}}
.
0
1"
trackable_list_wrapper
'
B0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�

trainable_variables
 layer_metrics
!non_trainable_variables
"layer_regularization_losses
regularization_losses
#metrics
	variables

$layers
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
�	

kernel
bias
%trainable_variables
&	variables
'regularization_losses
(	keras_api
C__call__
*D&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 115, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [115, 64]}}
.
0
1"
trackable_list_wrapper
'
E0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
trainable_variables
)layer_metrics
*non_trainable_variables
+layer_regularization_losses
regularization_losses
,metrics
	variables

-layers
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
:s@2dense/kernel
:@2
dense/bias
 :@s2dense_1/kernel
:s2dense_1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
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
�
trainable_variables
.layer_metrics
/non_trainable_variables
	variables
0layer_regularization_losses
regularization_losses
1metrics

2layers
@__call__
Factivity_regularizer_fn
*A&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
	0"
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
�
%trainable_variables
3layer_metrics
4non_trainable_variables
&	variables
5layer_regularization_losses
'regularization_losses
6metrics

7layers
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
!__inference__wrapped_model_920319�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!�
input_1���������s
�2�
,__inference_autoencoder_layer_call_fn_920697
,__inference_autoencoder_layer_call_fn_920864
,__inference_autoencoder_layer_call_fn_920878
,__inference_autoencoder_layer_call_fn_920767�
���
FullArgSpec$
args�
jself
jX

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_autoencoder_layer_call_and_return_conditional_losses_920938
G__inference_autoencoder_layer_call_and_return_conditional_losses_920998
G__inference_autoencoder_layer_call_and_return_conditional_losses_920795
G__inference_autoencoder_layer_call_and_return_conditional_losses_920823�
���
FullArgSpec$
args�
jself
jX

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_sequential_layer_call_fn_920403
+__inference_sequential_layer_call_fn_921014
+__inference_sequential_layer_call_fn_921024
+__inference_sequential_layer_call_fn_920479�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
F__inference_sequential_layer_call_and_return_conditional_losses_921071
F__inference_sequential_layer_call_and_return_conditional_losses_921118
F__inference_sequential_layer_call_and_return_conditional_losses_920503
F__inference_sequential_layer_call_and_return_conditional_losses_920527�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
-__inference_sequential_1_layer_call_fn_921133
-__inference_sequential_1_layer_call_fn_921142
-__inference_sequential_1_layer_call_fn_921151
-__inference_sequential_1_layer_call_fn_921160�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_sequential_1_layer_call_and_return_conditional_losses_921177
H__inference_sequential_1_layer_call_and_return_conditional_losses_921194
H__inference_sequential_1_layer_call_and_return_conditional_losses_921211
H__inference_sequential_1_layer_call_and_return_conditional_losses_921228�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
$__inference_signature_wrapper_920850input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_dense_layer_call_fn_921243�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_layer_call_and_return_all_conditional_losses_921254�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
__inference_loss_fn_0_921265�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
(__inference_dense_1_layer_call_fn_921280�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_1_layer_call_and_return_conditional_losses_921297�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
__inference_loss_fn_1_921308�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
-__inference_dense_activity_regularizer_920349�
���
FullArgSpec!
args�
jself
j
activation
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
	�
�2�
A__inference_dense_layer_call_and_return_conditional_losses_921325�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
!__inference__wrapped_model_920319m0�-
&�#
!�
input_1���������s
� "3�0
.
output_1"�
output_1���������s�
G__inference_autoencoder_layer_call_and_return_conditional_losses_920795q4�1
*�'
!�
input_1���������s
p 
� "3�0
�
0���������s
�
�	
1/0 �
G__inference_autoencoder_layer_call_and_return_conditional_losses_920823q4�1
*�'
!�
input_1���������s
p
� "3�0
�
0���������s
�
�	
1/0 �
G__inference_autoencoder_layer_call_and_return_conditional_losses_920938k.�+
$�!
�
X���������s
p 
� "3�0
�
0���������s
�
�	
1/0 �
G__inference_autoencoder_layer_call_and_return_conditional_losses_920998k.�+
$�!
�
X���������s
p
� "3�0
�
0���������s
�
�	
1/0 �
,__inference_autoencoder_layer_call_fn_920697V4�1
*�'
!�
input_1���������s
p 
� "����������s�
,__inference_autoencoder_layer_call_fn_920767V4�1
*�'
!�
input_1���������s
p
� "����������s�
,__inference_autoencoder_layer_call_fn_920864P.�+
$�!
�
X���������s
p 
� "����������s�
,__inference_autoencoder_layer_call_fn_920878P.�+
$�!
�
X���������s
p
� "����������s�
C__inference_dense_1_layer_call_and_return_conditional_losses_921297\/�,
%�"
 �
inputs���������@
� "%�"
�
0���������s
� {
(__inference_dense_1_layer_call_fn_921280O/�,
%�"
 �
inputs���������@
� "����������s`
-__inference_dense_activity_regularizer_920349/$�!
�
�

activation
� "� �
E__inference_dense_layer_call_and_return_all_conditional_losses_921254j/�,
%�"
 �
inputs���������s
� "3�0
�
0���������@
�
�	
1/0 �
A__inference_dense_layer_call_and_return_conditional_losses_921325\/�,
%�"
 �
inputs���������s
� "%�"
�
0���������@
� y
&__inference_dense_layer_call_fn_921243O/�,
%�"
 �
inputs���������s
� "����������@;
__inference_loss_fn_0_921265�

� 
� "� ;
__inference_loss_fn_1_921308�

� 
� "� �
H__inference_sequential_1_layer_call_and_return_conditional_losses_921177d7�4
-�*
 �
inputs���������@
p 

 
� "%�"
�
0���������s
� �
H__inference_sequential_1_layer_call_and_return_conditional_losses_921194d7�4
-�*
 �
inputs���������@
p

 
� "%�"
�
0���������s
� �
H__inference_sequential_1_layer_call_and_return_conditional_losses_921211k>�;
4�1
'�$
dense_1_input���������@
p 

 
� "%�"
�
0���������s
� �
H__inference_sequential_1_layer_call_and_return_conditional_losses_921228k>�;
4�1
'�$
dense_1_input���������@
p

 
� "%�"
�
0���������s
� �
-__inference_sequential_1_layer_call_fn_921133^>�;
4�1
'�$
dense_1_input���������@
p 

 
� "����������s�
-__inference_sequential_1_layer_call_fn_921142W7�4
-�*
 �
inputs���������@
p 

 
� "����������s�
-__inference_sequential_1_layer_call_fn_921151W7�4
-�*
 �
inputs���������@
p

 
� "����������s�
-__inference_sequential_1_layer_call_fn_921160^>�;
4�1
'�$
dense_1_input���������@
p

 
� "����������s�
F__inference_sequential_layer_call_and_return_conditional_losses_920503s8�5
.�+
!�
input_1���������s
p 

 
� "3�0
�
0���������@
�
�	
1/0 �
F__inference_sequential_layer_call_and_return_conditional_losses_920527s8�5
.�+
!�
input_1���������s
p

 
� "3�0
�
0���������@
�
�	
1/0 �
F__inference_sequential_layer_call_and_return_conditional_losses_921071r7�4
-�*
 �
inputs���������s
p 

 
� "3�0
�
0���������@
�
�	
1/0 �
F__inference_sequential_layer_call_and_return_conditional_losses_921118r7�4
-�*
 �
inputs���������s
p

 
� "3�0
�
0���������@
�
�	
1/0 �
+__inference_sequential_layer_call_fn_920403X8�5
.�+
!�
input_1���������s
p 

 
� "����������@�
+__inference_sequential_layer_call_fn_920479X8�5
.�+
!�
input_1���������s
p

 
� "����������@�
+__inference_sequential_layer_call_fn_921014W7�4
-�*
 �
inputs���������s
p 

 
� "����������@�
+__inference_sequential_layer_call_fn_921024W7�4
-�*
 �
inputs���������s
p

 
� "����������@�
$__inference_signature_wrapper_920850x;�8
� 
1�.
,
input_1!�
input_1���������s"3�0
.
output_1"�
output_1���������s