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
|
dense_108/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:^ *!
shared_namedense_108/kernel
u
$dense_108/kernel/Read/ReadVariableOpReadVariableOpdense_108/kernel*
_output_shapes

:^ *
dtype0
t
dense_108/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_108/bias
m
"dense_108/bias/Read/ReadVariableOpReadVariableOpdense_108/bias*
_output_shapes
: *
dtype0
|
dense_109/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: ^*!
shared_namedense_109/kernel
u
$dense_109/kernel/Read/ReadVariableOpReadVariableOpdense_109/kernel*
_output_shapes

: ^*
dtype0
t
dense_109/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:^*
shared_namedense_109/bias
m
"dense_109/bias/Read/ReadVariableOpReadVariableOpdense_109/bias*
_output_shapes
:^*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
history
encoder
decoder
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
y
	layer_with_weights-0
	layer-0

trainable_variables
	variables
regularization_losses
	keras_api
y
layer_with_weights-0
layer-0
trainable_variables
	variables
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
layer_regularization_losses
non_trainable_variables
metrics
trainable_variables

layers
	variables
layer_metrics
regularization_losses
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

0
1
 
?
 layer_regularization_losses
!non_trainable_variables
"metrics

trainable_variables

#layers
	variables
$layer_metrics
regularization_losses
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

0
1
 
?
)layer_regularization_losses
*non_trainable_variables
+metrics
trainable_variables

,layers
	variables
-layer_metrics
regularization_losses
VT
VARIABLE_VALUEdense_108/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_108/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_109/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_109/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
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
.layer_regularization_losses
/metrics
0non_trainable_variables
trainable_variables

1layers
	variables
2layer_metrics
regularization_losses
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
3layer_regularization_losses
4metrics
5non_trainable_variables
%trainable_variables

6layers
&	variables
7layer_metrics
'regularization_losses
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
:?????????^*
dtype0*
shape:?????????^
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_108/kerneldense_108/biasdense_109/kerneldense_109/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_16643771
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_108/kernel/Read/ReadVariableOp"dense_108/bias/Read/ReadVariableOp$dense_109/kernel/Read/ReadVariableOp"dense_109/bias/Read/ReadVariableOpConst*
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
GPU 2J 8? **
f%R#
!__inference__traced_save_16644277
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_108/kerneldense_108/biasdense_109/kerneldense_109/bias*
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
GPU 2J 8? *-
f(R&
$__inference__traced_restore_16644299??	
?%
?
L__inference_autoencoder_54_layer_call_and_return_conditional_losses_16643744
input_1)
sequential_108_16643719:^ %
sequential_108_16643721: )
sequential_109_16643725: ^%
sequential_109_16643727:^
identity

identity_1??2dense_108/kernel/Regularizer/Square/ReadVariableOp?2dense_109/kernel/Regularizer/Square/ReadVariableOp?&sequential_108/StatefulPartitionedCall?&sequential_109/StatefulPartitionedCall?
&sequential_108/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_108_16643719sequential_108_16643721*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_108_layer_call_and_return_conditional_losses_166433822(
&sequential_108/StatefulPartitionedCall?
&sequential_109/StatefulPartitionedCallStatefulPartitionedCall/sequential_108/StatefulPartitionedCall:output:0sequential_109_16643725sequential_109_16643727*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_109_layer_call_and_return_conditional_losses_166435282(
&sequential_109/StatefulPartitionedCall?
2dense_108/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_108_16643719*
_output_shapes

:^ *
dtype024
2dense_108/kernel/Regularizer/Square/ReadVariableOp?
#dense_108/kernel/Regularizer/SquareSquare:dense_108/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_108/kernel/Regularizer/Square?
"dense_108/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_108/kernel/Regularizer/Const?
 dense_108/kernel/Regularizer/SumSum'dense_108/kernel/Regularizer/Square:y:0+dense_108/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/Sum?
"dense_108/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_108/kernel/Regularizer/mul/x?
 dense_108/kernel/Regularizer/mulMul+dense_108/kernel/Regularizer/mul/x:output:0)dense_108/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/mul?
2dense_109/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_109_16643725*
_output_shapes

: ^*
dtype024
2dense_109/kernel/Regularizer/Square/ReadVariableOp?
#dense_109/kernel/Regularizer/SquareSquare:dense_109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_109/kernel/Regularizer/Square?
"dense_109/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_109/kernel/Regularizer/Const?
 dense_109/kernel/Regularizer/SumSum'dense_109/kernel/Regularizer/Square:y:0+dense_109/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/Sum?
"dense_109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_109/kernel/Regularizer/mul/x?
 dense_109/kernel/Regularizer/mulMul+dense_109/kernel/Regularizer/mul/x:output:0)dense_109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/mul?
IdentityIdentity/sequential_109/StatefulPartitionedCall:output:03^dense_108/kernel/Regularizer/Square/ReadVariableOp3^dense_109/kernel/Regularizer/Square/ReadVariableOp'^sequential_108/StatefulPartitionedCall'^sequential_109/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity/sequential_108/StatefulPartitionedCall:output:13^dense_108/kernel/Regularizer/Square/ReadVariableOp3^dense_109/kernel/Regularizer/Square/ReadVariableOp'^sequential_108/StatefulPartitionedCall'^sequential_109/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_108/kernel/Regularizer/Square/ReadVariableOp2dense_108/kernel/Regularizer/Square/ReadVariableOp2h
2dense_109/kernel/Regularizer/Square/ReadVariableOp2dense_109/kernel/Regularizer/Square/ReadVariableOp2P
&sequential_108/StatefulPartitionedCall&sequential_108/StatefulPartitionedCall2P
&sequential_109/StatefulPartitionedCall&sequential_109/StatefulPartitionedCall:P L
'
_output_shapes
:?????????^
!
_user_specified_name	input_1
?
?
1__inference_sequential_108_layer_call_fn_16643933

inputs
unknown:^ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_108_layer_call_and_return_conditional_losses_166433162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
L__inference_sequential_109_layer_call_and_return_conditional_losses_16643485

inputs$
dense_109_16643473: ^ 
dense_109_16643475:^
identity??!dense_109/StatefulPartitionedCall?2dense_109/kernel/Regularizer/Square/ReadVariableOp?
!dense_109/StatefulPartitionedCallStatefulPartitionedCallinputsdense_109_16643473dense_109_16643475*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_109_layer_call_and_return_conditional_losses_166434722#
!dense_109/StatefulPartitionedCall?
2dense_109/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_109_16643473*
_output_shapes

: ^*
dtype024
2dense_109/kernel/Regularizer/Square/ReadVariableOp?
#dense_109/kernel/Regularizer/SquareSquare:dense_109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_109/kernel/Regularizer/Square?
"dense_109/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_109/kernel/Regularizer/Const?
 dense_109/kernel/Regularizer/SumSum'dense_109/kernel/Regularizer/Square:y:0+dense_109/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/Sum?
"dense_109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_109/kernel/Regularizer/mul/x?
 dense_109/kernel/Regularizer/mulMul+dense_109/kernel/Regularizer/mul/x:output:0)dense_109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/mul?
IdentityIdentity*dense_109/StatefulPartitionedCall:output:0"^dense_109/StatefulPartitionedCall3^dense_109/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2F
!dense_109/StatefulPartitionedCall!dense_109/StatefulPartitionedCall2h
2dense_109/kernel/Regularizer/Square/ReadVariableOp2dense_109/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?_
?
#__inference__wrapped_model_16643241
input_1X
Fautoencoder_54_sequential_108_dense_108_matmul_readvariableop_resource:^ U
Gautoencoder_54_sequential_108_dense_108_biasadd_readvariableop_resource: X
Fautoencoder_54_sequential_109_dense_109_matmul_readvariableop_resource: ^U
Gautoencoder_54_sequential_109_dense_109_biasadd_readvariableop_resource:^
identity??>autoencoder_54/sequential_108/dense_108/BiasAdd/ReadVariableOp?=autoencoder_54/sequential_108/dense_108/MatMul/ReadVariableOp?>autoencoder_54/sequential_109/dense_109/BiasAdd/ReadVariableOp?=autoencoder_54/sequential_109/dense_109/MatMul/ReadVariableOp?
=autoencoder_54/sequential_108/dense_108/MatMul/ReadVariableOpReadVariableOpFautoencoder_54_sequential_108_dense_108_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02?
=autoencoder_54/sequential_108/dense_108/MatMul/ReadVariableOp?
.autoencoder_54/sequential_108/dense_108/MatMulMatMulinput_1Eautoencoder_54/sequential_108/dense_108/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 20
.autoencoder_54/sequential_108/dense_108/MatMul?
>autoencoder_54/sequential_108/dense_108/BiasAdd/ReadVariableOpReadVariableOpGautoencoder_54_sequential_108_dense_108_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02@
>autoencoder_54/sequential_108/dense_108/BiasAdd/ReadVariableOp?
/autoencoder_54/sequential_108/dense_108/BiasAddBiasAdd8autoencoder_54/sequential_108/dense_108/MatMul:product:0Fautoencoder_54/sequential_108/dense_108/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 21
/autoencoder_54/sequential_108/dense_108/BiasAdd?
/autoencoder_54/sequential_108/dense_108/SigmoidSigmoid8autoencoder_54/sequential_108/dense_108/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 21
/autoencoder_54/sequential_108/dense_108/Sigmoid?
Rautoencoder_54/sequential_108/dense_108/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2T
Rautoencoder_54/sequential_108/dense_108/ActivityRegularizer/Mean/reduction_indices?
@autoencoder_54/sequential_108/dense_108/ActivityRegularizer/MeanMean3autoencoder_54/sequential_108/dense_108/Sigmoid:y:0[autoencoder_54/sequential_108/dense_108/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2B
@autoencoder_54/sequential_108/dense_108/ActivityRegularizer/Mean?
Eautoencoder_54/sequential_108/dense_108/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2G
Eautoencoder_54/sequential_108/dense_108/ActivityRegularizer/Maximum/y?
Cautoencoder_54/sequential_108/dense_108/ActivityRegularizer/MaximumMaximumIautoencoder_54/sequential_108/dense_108/ActivityRegularizer/Mean:output:0Nautoencoder_54/sequential_108/dense_108/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2E
Cautoencoder_54/sequential_108/dense_108/ActivityRegularizer/Maximum?
Eautoencoder_54/sequential_108/dense_108/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2G
Eautoencoder_54/sequential_108/dense_108/ActivityRegularizer/truediv/x?
Cautoencoder_54/sequential_108/dense_108/ActivityRegularizer/truedivRealDivNautoencoder_54/sequential_108/dense_108/ActivityRegularizer/truediv/x:output:0Gautoencoder_54/sequential_108/dense_108/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2E
Cautoencoder_54/sequential_108/dense_108/ActivityRegularizer/truediv?
?autoencoder_54/sequential_108/dense_108/ActivityRegularizer/LogLogGautoencoder_54/sequential_108/dense_108/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2A
?autoencoder_54/sequential_108/dense_108/ActivityRegularizer/Log?
Aautoencoder_54/sequential_108/dense_108/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2C
Aautoencoder_54/sequential_108/dense_108/ActivityRegularizer/mul/x?
?autoencoder_54/sequential_108/dense_108/ActivityRegularizer/mulMulJautoencoder_54/sequential_108/dense_108/ActivityRegularizer/mul/x:output:0Cautoencoder_54/sequential_108/dense_108/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2A
?autoencoder_54/sequential_108/dense_108/ActivityRegularizer/mul?
Aautoencoder_54/sequential_108/dense_108/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2C
Aautoencoder_54/sequential_108/dense_108/ActivityRegularizer/sub/x?
?autoencoder_54/sequential_108/dense_108/ActivityRegularizer/subSubJautoencoder_54/sequential_108/dense_108/ActivityRegularizer/sub/x:output:0Gautoencoder_54/sequential_108/dense_108/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2A
?autoencoder_54/sequential_108/dense_108/ActivityRegularizer/sub?
Gautoencoder_54/sequential_108/dense_108/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2I
Gautoencoder_54/sequential_108/dense_108/ActivityRegularizer/truediv_1/x?
Eautoencoder_54/sequential_108/dense_108/ActivityRegularizer/truediv_1RealDivPautoencoder_54/sequential_108/dense_108/ActivityRegularizer/truediv_1/x:output:0Cautoencoder_54/sequential_108/dense_108/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2G
Eautoencoder_54/sequential_108/dense_108/ActivityRegularizer/truediv_1?
Aautoencoder_54/sequential_108/dense_108/ActivityRegularizer/Log_1LogIautoencoder_54/sequential_108/dense_108/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2C
Aautoencoder_54/sequential_108/dense_108/ActivityRegularizer/Log_1?
Cautoencoder_54/sequential_108/dense_108/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2E
Cautoencoder_54/sequential_108/dense_108/ActivityRegularizer/mul_1/x?
Aautoencoder_54/sequential_108/dense_108/ActivityRegularizer/mul_1MulLautoencoder_54/sequential_108/dense_108/ActivityRegularizer/mul_1/x:output:0Eautoencoder_54/sequential_108/dense_108/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2C
Aautoencoder_54/sequential_108/dense_108/ActivityRegularizer/mul_1?
?autoencoder_54/sequential_108/dense_108/ActivityRegularizer/addAddV2Cautoencoder_54/sequential_108/dense_108/ActivityRegularizer/mul:z:0Eautoencoder_54/sequential_108/dense_108/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2A
?autoencoder_54/sequential_108/dense_108/ActivityRegularizer/add?
Aautoencoder_54/sequential_108/dense_108/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Aautoencoder_54/sequential_108/dense_108/ActivityRegularizer/Const?
?autoencoder_54/sequential_108/dense_108/ActivityRegularizer/SumSumCautoencoder_54/sequential_108/dense_108/ActivityRegularizer/add:z:0Jautoencoder_54/sequential_108/dense_108/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2A
?autoencoder_54/sequential_108/dense_108/ActivityRegularizer/Sum?
Cautoencoder_54/sequential_108/dense_108/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2E
Cautoencoder_54/sequential_108/dense_108/ActivityRegularizer/mul_2/x?
Aautoencoder_54/sequential_108/dense_108/ActivityRegularizer/mul_2MulLautoencoder_54/sequential_108/dense_108/ActivityRegularizer/mul_2/x:output:0Hautoencoder_54/sequential_108/dense_108/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2C
Aautoencoder_54/sequential_108/dense_108/ActivityRegularizer/mul_2?
Aautoencoder_54/sequential_108/dense_108/ActivityRegularizer/ShapeShape3autoencoder_54/sequential_108/dense_108/Sigmoid:y:0*
T0*
_output_shapes
:2C
Aautoencoder_54/sequential_108/dense_108/ActivityRegularizer/Shape?
Oautoencoder_54/sequential_108/dense_108/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Q
Oautoencoder_54/sequential_108/dense_108/ActivityRegularizer/strided_slice/stack?
Qautoencoder_54/sequential_108/dense_108/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2S
Qautoencoder_54/sequential_108/dense_108/ActivityRegularizer/strided_slice/stack_1?
Qautoencoder_54/sequential_108/dense_108/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
Qautoencoder_54/sequential_108/dense_108/ActivityRegularizer/strided_slice/stack_2?
Iautoencoder_54/sequential_108/dense_108/ActivityRegularizer/strided_sliceStridedSliceJautoencoder_54/sequential_108/dense_108/ActivityRegularizer/Shape:output:0Xautoencoder_54/sequential_108/dense_108/ActivityRegularizer/strided_slice/stack:output:0Zautoencoder_54/sequential_108/dense_108/ActivityRegularizer/strided_slice/stack_1:output:0Zautoencoder_54/sequential_108/dense_108/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2K
Iautoencoder_54/sequential_108/dense_108/ActivityRegularizer/strided_slice?
@autoencoder_54/sequential_108/dense_108/ActivityRegularizer/CastCastRautoencoder_54/sequential_108/dense_108/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2B
@autoencoder_54/sequential_108/dense_108/ActivityRegularizer/Cast?
Eautoencoder_54/sequential_108/dense_108/ActivityRegularizer/truediv_2RealDivEautoencoder_54/sequential_108/dense_108/ActivityRegularizer/mul_2:z:0Dautoencoder_54/sequential_108/dense_108/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2G
Eautoencoder_54/sequential_108/dense_108/ActivityRegularizer/truediv_2?
=autoencoder_54/sequential_109/dense_109/MatMul/ReadVariableOpReadVariableOpFautoencoder_54_sequential_109_dense_109_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02?
=autoencoder_54/sequential_109/dense_109/MatMul/ReadVariableOp?
.autoencoder_54/sequential_109/dense_109/MatMulMatMul3autoencoder_54/sequential_108/dense_108/Sigmoid:y:0Eautoencoder_54/sequential_109/dense_109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^20
.autoencoder_54/sequential_109/dense_109/MatMul?
>autoencoder_54/sequential_109/dense_109/BiasAdd/ReadVariableOpReadVariableOpGautoencoder_54_sequential_109_dense_109_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02@
>autoencoder_54/sequential_109/dense_109/BiasAdd/ReadVariableOp?
/autoencoder_54/sequential_109/dense_109/BiasAddBiasAdd8autoencoder_54/sequential_109/dense_109/MatMul:product:0Fautoencoder_54/sequential_109/dense_109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^21
/autoencoder_54/sequential_109/dense_109/BiasAdd?
/autoencoder_54/sequential_109/dense_109/SigmoidSigmoid8autoencoder_54/sequential_109/dense_109/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^21
/autoencoder_54/sequential_109/dense_109/Sigmoid?
IdentityIdentity3autoencoder_54/sequential_109/dense_109/Sigmoid:y:0?^autoencoder_54/sequential_108/dense_108/BiasAdd/ReadVariableOp>^autoencoder_54/sequential_108/dense_108/MatMul/ReadVariableOp?^autoencoder_54/sequential_109/dense_109/BiasAdd/ReadVariableOp>^autoencoder_54/sequential_109/dense_109/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2?
>autoencoder_54/sequential_108/dense_108/BiasAdd/ReadVariableOp>autoencoder_54/sequential_108/dense_108/BiasAdd/ReadVariableOp2~
=autoencoder_54/sequential_108/dense_108/MatMul/ReadVariableOp=autoencoder_54/sequential_108/dense_108/MatMul/ReadVariableOp2?
>autoencoder_54/sequential_109/dense_109/BiasAdd/ReadVariableOp>autoencoder_54/sequential_109/dense_109/BiasAdd/ReadVariableOp2~
=autoencoder_54/sequential_109/dense_109/MatMul/ReadVariableOp=autoencoder_54/sequential_109/dense_109/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????^
!
_user_specified_name	input_1
?
?
1__inference_sequential_109_layer_call_fn_16644077
dense_109_input
unknown: ^
	unknown_0:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_109_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_109_layer_call_and_return_conditional_losses_166435282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:????????? 
)
_user_specified_namedense_109_input
?
?
$__inference__traced_restore_16644299
file_prefix3
!assignvariableop_dense_108_kernel:^ /
!assignvariableop_1_dense_108_bias: 5
#assignvariableop_2_dense_109_kernel: ^/
!assignvariableop_3_dense_109_bias:^

identity_5??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
AssignVariableOpAssignVariableOp!assignvariableop_dense_108_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_108_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_109_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_109_biasIdentity_3:output:0"/device:CPU:0*
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
?
S
3__inference_dense_108_activity_regularizer_16643270

activation
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
Mean/reduction_indicesd
MeanMean
activationMean/reduction_indices:output:0*
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
?
?
1__inference_autoencoder_54_layer_call_fn_16643799
x
unknown:^ 
	unknown_0: 
	unknown_1: ^
	unknown_2:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????^: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_autoencoder_54_layer_call_and_return_conditional_losses_166436622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?%
?
L__inference_autoencoder_54_layer_call_and_return_conditional_losses_16643716
input_1)
sequential_108_16643691:^ %
sequential_108_16643693: )
sequential_109_16643697: ^%
sequential_109_16643699:^
identity

identity_1??2dense_108/kernel/Regularizer/Square/ReadVariableOp?2dense_109/kernel/Regularizer/Square/ReadVariableOp?&sequential_108/StatefulPartitionedCall?&sequential_109/StatefulPartitionedCall?
&sequential_108/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_108_16643691sequential_108_16643693*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_108_layer_call_and_return_conditional_losses_166433162(
&sequential_108/StatefulPartitionedCall?
&sequential_109/StatefulPartitionedCallStatefulPartitionedCall/sequential_108/StatefulPartitionedCall:output:0sequential_109_16643697sequential_109_16643699*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_109_layer_call_and_return_conditional_losses_166434852(
&sequential_109/StatefulPartitionedCall?
2dense_108/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_108_16643691*
_output_shapes

:^ *
dtype024
2dense_108/kernel/Regularizer/Square/ReadVariableOp?
#dense_108/kernel/Regularizer/SquareSquare:dense_108/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_108/kernel/Regularizer/Square?
"dense_108/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_108/kernel/Regularizer/Const?
 dense_108/kernel/Regularizer/SumSum'dense_108/kernel/Regularizer/Square:y:0+dense_108/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/Sum?
"dense_108/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_108/kernel/Regularizer/mul/x?
 dense_108/kernel/Regularizer/mulMul+dense_108/kernel/Regularizer/mul/x:output:0)dense_108/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/mul?
2dense_109/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_109_16643697*
_output_shapes

: ^*
dtype024
2dense_109/kernel/Regularizer/Square/ReadVariableOp?
#dense_109/kernel/Regularizer/SquareSquare:dense_109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_109/kernel/Regularizer/Square?
"dense_109/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_109/kernel/Regularizer/Const?
 dense_109/kernel/Regularizer/SumSum'dense_109/kernel/Regularizer/Square:y:0+dense_109/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/Sum?
"dense_109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_109/kernel/Regularizer/mul/x?
 dense_109/kernel/Regularizer/mulMul+dense_109/kernel/Regularizer/mul/x:output:0)dense_109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/mul?
IdentityIdentity/sequential_109/StatefulPartitionedCall:output:03^dense_108/kernel/Regularizer/Square/ReadVariableOp3^dense_109/kernel/Regularizer/Square/ReadVariableOp'^sequential_108/StatefulPartitionedCall'^sequential_109/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity/sequential_108/StatefulPartitionedCall:output:13^dense_108/kernel/Regularizer/Square/ReadVariableOp3^dense_109/kernel/Regularizer/Square/ReadVariableOp'^sequential_108/StatefulPartitionedCall'^sequential_109/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_108/kernel/Regularizer/Square/ReadVariableOp2dense_108/kernel/Regularizer/Square/ReadVariableOp2h
2dense_109/kernel/Regularizer/Square/ReadVariableOp2dense_109/kernel/Regularizer/Square/ReadVariableOp2P
&sequential_108/StatefulPartitionedCall&sequential_108/StatefulPartitionedCall2P
&sequential_109/StatefulPartitionedCall&sequential_109/StatefulPartitionedCall:P L
'
_output_shapes
:?????????^
!
_user_specified_name	input_1
?
?
1__inference_autoencoder_54_layer_call_fn_16643785
x
unknown:^ 
	unknown_0: 
	unknown_1: ^
	unknown_2:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????^: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_autoencoder_54_layer_call_and_return_conditional_losses_166436062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?
?
,__inference_dense_109_layer_call_fn_16644197

inputs
unknown: ^
	unknown_0:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_109_layer_call_and_return_conditional_losses_166434722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

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
1__inference_sequential_109_layer_call_fn_16644050
dense_109_input
unknown: ^
	unknown_0:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_109_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_109_layer_call_and_return_conditional_losses_166434852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:????????? 
)
_user_specified_namedense_109_input
?
?
1__inference_sequential_109_layer_call_fn_16644068

inputs
unknown: ^
	unknown_0:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_109_layer_call_and_return_conditional_losses_166435282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

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
?#
?
L__inference_sequential_108_layer_call_and_return_conditional_losses_16643316

inputs$
dense_108_16643295:^  
dense_108_16643297: 
identity

identity_1??!dense_108/StatefulPartitionedCall?2dense_108/kernel/Regularizer/Square/ReadVariableOp?
!dense_108/StatefulPartitionedCallStatefulPartitionedCallinputsdense_108_16643295dense_108_16643297*
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
GPU 2J 8? *P
fKRI
G__inference_dense_108_layer_call_and_return_conditional_losses_166432942#
!dense_108/StatefulPartitionedCall?
-dense_108/ActivityRegularizer/PartitionedCallPartitionedCall*dense_108/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *<
f7R5
3__inference_dense_108_activity_regularizer_166432702/
-dense_108/ActivityRegularizer/PartitionedCall?
#dense_108/ActivityRegularizer/ShapeShape*dense_108/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2%
#dense_108/ActivityRegularizer/Shape?
1dense_108/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_108/ActivityRegularizer/strided_slice/stack?
3dense_108/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_108/ActivityRegularizer/strided_slice/stack_1?
3dense_108/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_108/ActivityRegularizer/strided_slice/stack_2?
+dense_108/ActivityRegularizer/strided_sliceStridedSlice,dense_108/ActivityRegularizer/Shape:output:0:dense_108/ActivityRegularizer/strided_slice/stack:output:0<dense_108/ActivityRegularizer/strided_slice/stack_1:output:0<dense_108/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_108/ActivityRegularizer/strided_slice?
"dense_108/ActivityRegularizer/CastCast4dense_108/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_108/ActivityRegularizer/Cast?
%dense_108/ActivityRegularizer/truedivRealDiv6dense_108/ActivityRegularizer/PartitionedCall:output:0&dense_108/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_108/ActivityRegularizer/truediv?
2dense_108/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_108_16643295*
_output_shapes

:^ *
dtype024
2dense_108/kernel/Regularizer/Square/ReadVariableOp?
#dense_108/kernel/Regularizer/SquareSquare:dense_108/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_108/kernel/Regularizer/Square?
"dense_108/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_108/kernel/Regularizer/Const?
 dense_108/kernel/Regularizer/SumSum'dense_108/kernel/Regularizer/Square:y:0+dense_108/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/Sum?
"dense_108/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_108/kernel/Regularizer/mul/x?
 dense_108/kernel/Regularizer/mulMul+dense_108/kernel/Regularizer/mul/x:output:0)dense_108/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/mul?
IdentityIdentity*dense_108/StatefulPartitionedCall:output:0"^dense_108/StatefulPartitionedCall3^dense_108/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity)dense_108/ActivityRegularizer/truediv:z:0"^dense_108/StatefulPartitionedCall3^dense_108/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2F
!dense_108/StatefulPartitionedCall!dense_108/StatefulPartitionedCall2h
2dense_108/kernel/Regularizer/Square/ReadVariableOp2dense_108/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
L__inference_sequential_109_layer_call_and_return_conditional_losses_16644111

inputs:
(dense_109_matmul_readvariableop_resource: ^7
)dense_109_biasadd_readvariableop_resource:^
identity?? dense_109/BiasAdd/ReadVariableOp?dense_109/MatMul/ReadVariableOp?2dense_109/kernel/Regularizer/Square/ReadVariableOp?
dense_109/MatMul/ReadVariableOpReadVariableOp(dense_109_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02!
dense_109/MatMul/ReadVariableOp?
dense_109/MatMulMatMulinputs'dense_109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_109/MatMul?
 dense_109/BiasAdd/ReadVariableOpReadVariableOp)dense_109_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02"
 dense_109/BiasAdd/ReadVariableOp?
dense_109/BiasAddBiasAdddense_109/MatMul:product:0(dense_109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_109/BiasAdd
dense_109/SigmoidSigmoiddense_109/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2
dense_109/Sigmoid?
2dense_109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_109_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_109/kernel/Regularizer/Square/ReadVariableOp?
#dense_109/kernel/Regularizer/SquareSquare:dense_109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_109/kernel/Regularizer/Square?
"dense_109/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_109/kernel/Regularizer/Const?
 dense_109/kernel/Regularizer/SumSum'dense_109/kernel/Regularizer/Square:y:0+dense_109/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/Sum?
"dense_109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_109/kernel/Regularizer/mul/x?
 dense_109/kernel/Regularizer/mulMul+dense_109/kernel/Regularizer/mul/x:output:0)dense_109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/mul?
IdentityIdentitydense_109/Sigmoid:y:0!^dense_109/BiasAdd/ReadVariableOp ^dense_109/MatMul/ReadVariableOp3^dense_109/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2D
 dense_109/BiasAdd/ReadVariableOp dense_109/BiasAdd/ReadVariableOp2B
dense_109/MatMul/ReadVariableOpdense_109/MatMul/ReadVariableOp2h
2dense_109/kernel/Regularizer/Square/ReadVariableOp2dense_109/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
G__inference_dense_108_layer_call_and_return_conditional_losses_16644242

inputs0
matmul_readvariableop_resource:^ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_108/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
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
2dense_108/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_108/kernel/Regularizer/Square/ReadVariableOp?
#dense_108/kernel/Regularizer/SquareSquare:dense_108/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_108/kernel/Regularizer/Square?
"dense_108/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_108/kernel/Regularizer/Const?
 dense_108/kernel/Regularizer/SumSum'dense_108/kernel/Regularizer/Square:y:0+dense_108/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/Sum?
"dense_108/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_108/kernel/Regularizer/mul/x?
 dense_108/kernel/Regularizer/mulMul+dense_108/kernel/Regularizer/mul/x:output:0)dense_108/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_108/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_108/kernel/Regularizer/Square/ReadVariableOp2dense_108/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
1__inference_sequential_108_layer_call_fn_16643324
input_55
unknown:^ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_55unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_108_layer_call_and_return_conditional_losses_166433162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????^
"
_user_specified_name
input_55
?
?
L__inference_sequential_109_layer_call_and_return_conditional_losses_16644128
dense_109_input:
(dense_109_matmul_readvariableop_resource: ^7
)dense_109_biasadd_readvariableop_resource:^
identity?? dense_109/BiasAdd/ReadVariableOp?dense_109/MatMul/ReadVariableOp?2dense_109/kernel/Regularizer/Square/ReadVariableOp?
dense_109/MatMul/ReadVariableOpReadVariableOp(dense_109_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02!
dense_109/MatMul/ReadVariableOp?
dense_109/MatMulMatMuldense_109_input'dense_109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_109/MatMul?
 dense_109/BiasAdd/ReadVariableOpReadVariableOp)dense_109_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02"
 dense_109/BiasAdd/ReadVariableOp?
dense_109/BiasAddBiasAdddense_109/MatMul:product:0(dense_109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_109/BiasAdd
dense_109/SigmoidSigmoiddense_109/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2
dense_109/Sigmoid?
2dense_109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_109_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_109/kernel/Regularizer/Square/ReadVariableOp?
#dense_109/kernel/Regularizer/SquareSquare:dense_109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_109/kernel/Regularizer/Square?
"dense_109/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_109/kernel/Regularizer/Const?
 dense_109/kernel/Regularizer/SumSum'dense_109/kernel/Regularizer/Square:y:0+dense_109/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/Sum?
"dense_109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_109/kernel/Regularizer/mul/x?
 dense_109/kernel/Regularizer/mulMul+dense_109/kernel/Regularizer/mul/x:output:0)dense_109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/mul?
IdentityIdentitydense_109/Sigmoid:y:0!^dense_109/BiasAdd/ReadVariableOp ^dense_109/MatMul/ReadVariableOp3^dense_109/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2D
 dense_109/BiasAdd/ReadVariableOp dense_109/BiasAdd/ReadVariableOp2B
dense_109/MatMul/ReadVariableOpdense_109/MatMul/ReadVariableOp2h
2dense_109/kernel/Regularizer/Square/ReadVariableOp2dense_109/kernel/Regularizer/Square/ReadVariableOp:X T
'
_output_shapes
:????????? 
)
_user_specified_namedense_109_input
?
?
G__inference_dense_108_layer_call_and_return_conditional_losses_16643294

inputs0
matmul_readvariableop_resource:^ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_108/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
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
2dense_108/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_108/kernel/Regularizer/Square/ReadVariableOp?
#dense_108/kernel/Regularizer/SquareSquare:dense_108/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_108/kernel/Regularizer/Square?
"dense_108/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_108/kernel/Regularizer/Const?
 dense_108/kernel/Regularizer/SumSum'dense_108/kernel/Regularizer/Square:y:0+dense_108/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/Sum?
"dense_108/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_108/kernel/Regularizer/mul/x?
 dense_108/kernel/Regularizer/mulMul+dense_108/kernel/Regularizer/mul/x:output:0)dense_108/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_108/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_108/kernel/Regularizer/Square/ReadVariableOp2dense_108/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?#
?
L__inference_sequential_108_layer_call_and_return_conditional_losses_16643382

inputs$
dense_108_16643361:^  
dense_108_16643363: 
identity

identity_1??!dense_108/StatefulPartitionedCall?2dense_108/kernel/Regularizer/Square/ReadVariableOp?
!dense_108/StatefulPartitionedCallStatefulPartitionedCallinputsdense_108_16643361dense_108_16643363*
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
GPU 2J 8? *P
fKRI
G__inference_dense_108_layer_call_and_return_conditional_losses_166432942#
!dense_108/StatefulPartitionedCall?
-dense_108/ActivityRegularizer/PartitionedCallPartitionedCall*dense_108/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *<
f7R5
3__inference_dense_108_activity_regularizer_166432702/
-dense_108/ActivityRegularizer/PartitionedCall?
#dense_108/ActivityRegularizer/ShapeShape*dense_108/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2%
#dense_108/ActivityRegularizer/Shape?
1dense_108/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_108/ActivityRegularizer/strided_slice/stack?
3dense_108/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_108/ActivityRegularizer/strided_slice/stack_1?
3dense_108/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_108/ActivityRegularizer/strided_slice/stack_2?
+dense_108/ActivityRegularizer/strided_sliceStridedSlice,dense_108/ActivityRegularizer/Shape:output:0:dense_108/ActivityRegularizer/strided_slice/stack:output:0<dense_108/ActivityRegularizer/strided_slice/stack_1:output:0<dense_108/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_108/ActivityRegularizer/strided_slice?
"dense_108/ActivityRegularizer/CastCast4dense_108/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_108/ActivityRegularizer/Cast?
%dense_108/ActivityRegularizer/truedivRealDiv6dense_108/ActivityRegularizer/PartitionedCall:output:0&dense_108/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_108/ActivityRegularizer/truediv?
2dense_108/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_108_16643361*
_output_shapes

:^ *
dtype024
2dense_108/kernel/Regularizer/Square/ReadVariableOp?
#dense_108/kernel/Regularizer/SquareSquare:dense_108/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_108/kernel/Regularizer/Square?
"dense_108/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_108/kernel/Regularizer/Const?
 dense_108/kernel/Regularizer/SumSum'dense_108/kernel/Regularizer/Square:y:0+dense_108/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/Sum?
"dense_108/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_108/kernel/Regularizer/mul/x?
 dense_108/kernel/Regularizer/mulMul+dense_108/kernel/Regularizer/mul/x:output:0)dense_108/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/mul?
IdentityIdentity*dense_108/StatefulPartitionedCall:output:0"^dense_108/StatefulPartitionedCall3^dense_108/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity)dense_108/ActivityRegularizer/truediv:z:0"^dense_108/StatefulPartitionedCall3^dense_108/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2F
!dense_108/StatefulPartitionedCall!dense_108/StatefulPartitionedCall2h
2dense_108/kernel/Regularizer/Square/ReadVariableOp2dense_108/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
1__inference_sequential_108_layer_call_fn_16643400
input_55
unknown:^ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_55unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_108_layer_call_and_return_conditional_losses_166433822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????^
"
_user_specified_name
input_55
?
?
1__inference_autoencoder_54_layer_call_fn_16643688
input_1
unknown:^ 
	unknown_0: 
	unknown_1: ^
	unknown_2:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????^: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_autoencoder_54_layer_call_and_return_conditional_losses_166436622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????^
!
_user_specified_name	input_1
?h
?
L__inference_autoencoder_54_layer_call_and_return_conditional_losses_16643858
xI
7sequential_108_dense_108_matmul_readvariableop_resource:^ F
8sequential_108_dense_108_biasadd_readvariableop_resource: I
7sequential_109_dense_109_matmul_readvariableop_resource: ^F
8sequential_109_dense_109_biasadd_readvariableop_resource:^
identity

identity_1??2dense_108/kernel/Regularizer/Square/ReadVariableOp?2dense_109/kernel/Regularizer/Square/ReadVariableOp?/sequential_108/dense_108/BiasAdd/ReadVariableOp?.sequential_108/dense_108/MatMul/ReadVariableOp?/sequential_109/dense_109/BiasAdd/ReadVariableOp?.sequential_109/dense_109/MatMul/ReadVariableOp?
.sequential_108/dense_108/MatMul/ReadVariableOpReadVariableOp7sequential_108_dense_108_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype020
.sequential_108/dense_108/MatMul/ReadVariableOp?
sequential_108/dense_108/MatMulMatMulx6sequential_108/dense_108/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2!
sequential_108/dense_108/MatMul?
/sequential_108/dense_108/BiasAdd/ReadVariableOpReadVariableOp8sequential_108_dense_108_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential_108/dense_108/BiasAdd/ReadVariableOp?
 sequential_108/dense_108/BiasAddBiasAdd)sequential_108/dense_108/MatMul:product:07sequential_108/dense_108/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2"
 sequential_108/dense_108/BiasAdd?
 sequential_108/dense_108/SigmoidSigmoid)sequential_108/dense_108/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2"
 sequential_108/dense_108/Sigmoid?
Csequential_108/dense_108/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2E
Csequential_108/dense_108/ActivityRegularizer/Mean/reduction_indices?
1sequential_108/dense_108/ActivityRegularizer/MeanMean$sequential_108/dense_108/Sigmoid:y:0Lsequential_108/dense_108/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 23
1sequential_108/dense_108/ActivityRegularizer/Mean?
6sequential_108/dense_108/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.28
6sequential_108/dense_108/ActivityRegularizer/Maximum/y?
4sequential_108/dense_108/ActivityRegularizer/MaximumMaximum:sequential_108/dense_108/ActivityRegularizer/Mean:output:0?sequential_108/dense_108/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 26
4sequential_108/dense_108/ActivityRegularizer/Maximum?
6sequential_108/dense_108/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<28
6sequential_108/dense_108/ActivityRegularizer/truediv/x?
4sequential_108/dense_108/ActivityRegularizer/truedivRealDiv?sequential_108/dense_108/ActivityRegularizer/truediv/x:output:08sequential_108/dense_108/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 26
4sequential_108/dense_108/ActivityRegularizer/truediv?
0sequential_108/dense_108/ActivityRegularizer/LogLog8sequential_108/dense_108/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 22
0sequential_108/dense_108/ActivityRegularizer/Log?
2sequential_108/dense_108/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<24
2sequential_108/dense_108/ActivityRegularizer/mul/x?
0sequential_108/dense_108/ActivityRegularizer/mulMul;sequential_108/dense_108/ActivityRegularizer/mul/x:output:04sequential_108/dense_108/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 22
0sequential_108/dense_108/ActivityRegularizer/mul?
2sequential_108/dense_108/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2sequential_108/dense_108/ActivityRegularizer/sub/x?
0sequential_108/dense_108/ActivityRegularizer/subSub;sequential_108/dense_108/ActivityRegularizer/sub/x:output:08sequential_108/dense_108/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 22
0sequential_108/dense_108/ActivityRegularizer/sub?
8sequential_108/dense_108/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2:
8sequential_108/dense_108/ActivityRegularizer/truediv_1/x?
6sequential_108/dense_108/ActivityRegularizer/truediv_1RealDivAsequential_108/dense_108/ActivityRegularizer/truediv_1/x:output:04sequential_108/dense_108/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 28
6sequential_108/dense_108/ActivityRegularizer/truediv_1?
2sequential_108/dense_108/ActivityRegularizer/Log_1Log:sequential_108/dense_108/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 24
2sequential_108/dense_108/ActivityRegularizer/Log_1?
4sequential_108/dense_108/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?26
4sequential_108/dense_108/ActivityRegularizer/mul_1/x?
2sequential_108/dense_108/ActivityRegularizer/mul_1Mul=sequential_108/dense_108/ActivityRegularizer/mul_1/x:output:06sequential_108/dense_108/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 24
2sequential_108/dense_108/ActivityRegularizer/mul_1?
0sequential_108/dense_108/ActivityRegularizer/addAddV24sequential_108/dense_108/ActivityRegularizer/mul:z:06sequential_108/dense_108/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 22
0sequential_108/dense_108/ActivityRegularizer/add?
2sequential_108/dense_108/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 24
2sequential_108/dense_108/ActivityRegularizer/Const?
0sequential_108/dense_108/ActivityRegularizer/SumSum4sequential_108/dense_108/ActivityRegularizer/add:z:0;sequential_108/dense_108/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 22
0sequential_108/dense_108/ActivityRegularizer/Sum?
4sequential_108/dense_108/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??26
4sequential_108/dense_108/ActivityRegularizer/mul_2/x?
2sequential_108/dense_108/ActivityRegularizer/mul_2Mul=sequential_108/dense_108/ActivityRegularizer/mul_2/x:output:09sequential_108/dense_108/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 24
2sequential_108/dense_108/ActivityRegularizer/mul_2?
2sequential_108/dense_108/ActivityRegularizer/ShapeShape$sequential_108/dense_108/Sigmoid:y:0*
T0*
_output_shapes
:24
2sequential_108/dense_108/ActivityRegularizer/Shape?
@sequential_108/dense_108/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential_108/dense_108/ActivityRegularizer/strided_slice/stack?
Bsequential_108/dense_108/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_108/dense_108/ActivityRegularizer/strided_slice/stack_1?
Bsequential_108/dense_108/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_108/dense_108/ActivityRegularizer/strided_slice/stack_2?
:sequential_108/dense_108/ActivityRegularizer/strided_sliceStridedSlice;sequential_108/dense_108/ActivityRegularizer/Shape:output:0Isequential_108/dense_108/ActivityRegularizer/strided_slice/stack:output:0Ksequential_108/dense_108/ActivityRegularizer/strided_slice/stack_1:output:0Ksequential_108/dense_108/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_108/dense_108/ActivityRegularizer/strided_slice?
1sequential_108/dense_108/ActivityRegularizer/CastCastCsequential_108/dense_108/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 23
1sequential_108/dense_108/ActivityRegularizer/Cast?
6sequential_108/dense_108/ActivityRegularizer/truediv_2RealDiv6sequential_108/dense_108/ActivityRegularizer/mul_2:z:05sequential_108/dense_108/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 28
6sequential_108/dense_108/ActivityRegularizer/truediv_2?
.sequential_109/dense_109/MatMul/ReadVariableOpReadVariableOp7sequential_109_dense_109_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype020
.sequential_109/dense_109/MatMul/ReadVariableOp?
sequential_109/dense_109/MatMulMatMul$sequential_108/dense_108/Sigmoid:y:06sequential_109/dense_109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2!
sequential_109/dense_109/MatMul?
/sequential_109/dense_109/BiasAdd/ReadVariableOpReadVariableOp8sequential_109_dense_109_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype021
/sequential_109/dense_109/BiasAdd/ReadVariableOp?
 sequential_109/dense_109/BiasAddBiasAdd)sequential_109/dense_109/MatMul:product:07sequential_109/dense_109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2"
 sequential_109/dense_109/BiasAdd?
 sequential_109/dense_109/SigmoidSigmoid)sequential_109/dense_109/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2"
 sequential_109/dense_109/Sigmoid?
2dense_108/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7sequential_108_dense_108_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_108/kernel/Regularizer/Square/ReadVariableOp?
#dense_108/kernel/Regularizer/SquareSquare:dense_108/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_108/kernel/Regularizer/Square?
"dense_108/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_108/kernel/Regularizer/Const?
 dense_108/kernel/Regularizer/SumSum'dense_108/kernel/Regularizer/Square:y:0+dense_108/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/Sum?
"dense_108/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_108/kernel/Regularizer/mul/x?
 dense_108/kernel/Regularizer/mulMul+dense_108/kernel/Regularizer/mul/x:output:0)dense_108/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/mul?
2dense_109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7sequential_109_dense_109_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_109/kernel/Regularizer/Square/ReadVariableOp?
#dense_109/kernel/Regularizer/SquareSquare:dense_109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_109/kernel/Regularizer/Square?
"dense_109/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_109/kernel/Regularizer/Const?
 dense_109/kernel/Regularizer/SumSum'dense_109/kernel/Regularizer/Square:y:0+dense_109/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/Sum?
"dense_109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_109/kernel/Regularizer/mul/x?
 dense_109/kernel/Regularizer/mulMul+dense_109/kernel/Regularizer/mul/x:output:0)dense_109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/mul?
IdentityIdentity$sequential_109/dense_109/Sigmoid:y:03^dense_108/kernel/Regularizer/Square/ReadVariableOp3^dense_109/kernel/Regularizer/Square/ReadVariableOp0^sequential_108/dense_108/BiasAdd/ReadVariableOp/^sequential_108/dense_108/MatMul/ReadVariableOp0^sequential_109/dense_109/BiasAdd/ReadVariableOp/^sequential_109/dense_109/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity:sequential_108/dense_108/ActivityRegularizer/truediv_2:z:03^dense_108/kernel/Regularizer/Square/ReadVariableOp3^dense_109/kernel/Regularizer/Square/ReadVariableOp0^sequential_108/dense_108/BiasAdd/ReadVariableOp/^sequential_108/dense_108/MatMul/ReadVariableOp0^sequential_109/dense_109/BiasAdd/ReadVariableOp/^sequential_109/dense_109/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_108/kernel/Regularizer/Square/ReadVariableOp2dense_108/kernel/Regularizer/Square/ReadVariableOp2h
2dense_109/kernel/Regularizer/Square/ReadVariableOp2dense_109/kernel/Regularizer/Square/ReadVariableOp2b
/sequential_108/dense_108/BiasAdd/ReadVariableOp/sequential_108/dense_108/BiasAdd/ReadVariableOp2`
.sequential_108/dense_108/MatMul/ReadVariableOp.sequential_108/dense_108/MatMul/ReadVariableOp2b
/sequential_109/dense_109/BiasAdd/ReadVariableOp/sequential_109/dense_109/BiasAdd/ReadVariableOp2`
.sequential_109/dense_109/MatMul/ReadVariableOp.sequential_109/dense_109/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?
?
!__inference__traced_save_16644277
file_prefix/
+savev2_dense_108_kernel_read_readvariableop-
)savev2_dense_108_bias_read_readvariableop/
+savev2_dense_109_kernel_read_readvariableop-
)savev2_dense_109_bias_read_readvariableop
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
value?B?B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_108_kernel_read_readvariableop)savev2_dense_108_bias_read_readvariableop+savev2_dense_109_kernel_read_readvariableop)savev2_dense_109_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
$: :^ : : ^:^: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:^ : 

_output_shapes
: :$ 

_output_shapes

: ^: 

_output_shapes
:^:

_output_shapes
: 
?
?
G__inference_dense_109_layer_call_and_return_conditional_losses_16643472

inputs0
matmul_readvariableop_resource: ^-
biasadd_readvariableop_resource:^
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_109/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:^*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????^2	
Sigmoid?
2dense_109/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_109/kernel/Regularizer/Square/ReadVariableOp?
#dense_109/kernel/Regularizer/SquareSquare:dense_109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_109/kernel/Regularizer/Square?
"dense_109/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_109/kernel/Regularizer/Const?
 dense_109/kernel/Regularizer/SumSum'dense_109/kernel/Regularizer/Square:y:0+dense_109/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/Sum?
"dense_109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_109/kernel/Regularizer/mul/x?
 dense_109/kernel/Regularizer/mulMul+dense_109/kernel/Regularizer/mul/x:output:0)dense_109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_109/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_109/kernel/Regularizer/Square/ReadVariableOp2dense_109/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
L__inference_sequential_109_layer_call_and_return_conditional_losses_16643528

inputs$
dense_109_16643516: ^ 
dense_109_16643518:^
identity??!dense_109/StatefulPartitionedCall?2dense_109/kernel/Regularizer/Square/ReadVariableOp?
!dense_109/StatefulPartitionedCallStatefulPartitionedCallinputsdense_109_16643516dense_109_16643518*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_109_layer_call_and_return_conditional_losses_166434722#
!dense_109/StatefulPartitionedCall?
2dense_109/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_109_16643516*
_output_shapes

: ^*
dtype024
2dense_109/kernel/Regularizer/Square/ReadVariableOp?
#dense_109/kernel/Regularizer/SquareSquare:dense_109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_109/kernel/Regularizer/Square?
"dense_109/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_109/kernel/Regularizer/Const?
 dense_109/kernel/Regularizer/SumSum'dense_109/kernel/Regularizer/Square:y:0+dense_109/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/Sum?
"dense_109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_109/kernel/Regularizer/mul/x?
 dense_109/kernel/Regularizer/mulMul+dense_109/kernel/Regularizer/mul/x:output:0)dense_109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/mul?
IdentityIdentity*dense_109/StatefulPartitionedCall:output:0"^dense_109/StatefulPartitionedCall3^dense_109/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2F
!dense_109/StatefulPartitionedCall!dense_109/StatefulPartitionedCall2h
2dense_109/kernel/Regularizer/Square/ReadVariableOp2dense_109/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_16643771
input_1
unknown:^ 
	unknown_0: 
	unknown_1: ^
	unknown_2:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_166432412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????^
!
_user_specified_name	input_1
?
?
__inference_loss_fn_1_16644225M
;dense_109_kernel_regularizer_square_readvariableop_resource: ^
identity??2dense_109/kernel/Regularizer/Square/ReadVariableOp?
2dense_109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;dense_109_kernel_regularizer_square_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_109/kernel/Regularizer/Square/ReadVariableOp?
#dense_109/kernel/Regularizer/SquareSquare:dense_109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_109/kernel/Regularizer/Square?
"dense_109/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_109/kernel/Regularizer/Const?
 dense_109/kernel/Regularizer/SumSum'dense_109/kernel/Regularizer/Square:y:0+dense_109/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/Sum?
"dense_109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_109/kernel/Regularizer/mul/x?
 dense_109/kernel/Regularizer/mulMul+dense_109/kernel/Regularizer/mul/x:output:0)dense_109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/mul?
IdentityIdentity$dense_109/kernel/Regularizer/mul:z:03^dense_109/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_109/kernel/Regularizer/Square/ReadVariableOp2dense_109/kernel/Regularizer/Square/ReadVariableOp
?
?
K__inference_dense_108_layer_call_and_return_all_conditional_losses_16644171

inputs
unknown:^ 
	unknown_0: 
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
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dense_108_layer_call_and_return_conditional_losses_166432942
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
GPU 2J 8? *<
f7R5
3__inference_dense_108_activity_regularizer_166432702
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

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
:?????????^: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
G__inference_dense_109_layer_call_and_return_conditional_losses_16644214

inputs0
matmul_readvariableop_resource: ^-
biasadd_readvariableop_resource:^
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_109/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:^*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????^2	
Sigmoid?
2dense_109/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_109/kernel/Regularizer/Square/ReadVariableOp?
#dense_109/kernel/Regularizer/SquareSquare:dense_109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_109/kernel/Regularizer/Square?
"dense_109/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_109/kernel/Regularizer/Const?
 dense_109/kernel/Regularizer/SumSum'dense_109/kernel/Regularizer/Square:y:0+dense_109/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/Sum?
"dense_109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_109/kernel/Regularizer/mul/x?
 dense_109/kernel/Regularizer/mulMul+dense_109/kernel/Regularizer/mul/x:output:0)dense_109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_109/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2dense_109/kernel/Regularizer/Square/ReadVariableOp2dense_109/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?#
?
L__inference_sequential_108_layer_call_and_return_conditional_losses_16643448
input_55$
dense_108_16643427:^  
dense_108_16643429: 
identity

identity_1??!dense_108/StatefulPartitionedCall?2dense_108/kernel/Regularizer/Square/ReadVariableOp?
!dense_108/StatefulPartitionedCallStatefulPartitionedCallinput_55dense_108_16643427dense_108_16643429*
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
GPU 2J 8? *P
fKRI
G__inference_dense_108_layer_call_and_return_conditional_losses_166432942#
!dense_108/StatefulPartitionedCall?
-dense_108/ActivityRegularizer/PartitionedCallPartitionedCall*dense_108/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *<
f7R5
3__inference_dense_108_activity_regularizer_166432702/
-dense_108/ActivityRegularizer/PartitionedCall?
#dense_108/ActivityRegularizer/ShapeShape*dense_108/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2%
#dense_108/ActivityRegularizer/Shape?
1dense_108/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_108/ActivityRegularizer/strided_slice/stack?
3dense_108/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_108/ActivityRegularizer/strided_slice/stack_1?
3dense_108/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_108/ActivityRegularizer/strided_slice/stack_2?
+dense_108/ActivityRegularizer/strided_sliceStridedSlice,dense_108/ActivityRegularizer/Shape:output:0:dense_108/ActivityRegularizer/strided_slice/stack:output:0<dense_108/ActivityRegularizer/strided_slice/stack_1:output:0<dense_108/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_108/ActivityRegularizer/strided_slice?
"dense_108/ActivityRegularizer/CastCast4dense_108/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_108/ActivityRegularizer/Cast?
%dense_108/ActivityRegularizer/truedivRealDiv6dense_108/ActivityRegularizer/PartitionedCall:output:0&dense_108/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_108/ActivityRegularizer/truediv?
2dense_108/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_108_16643427*
_output_shapes

:^ *
dtype024
2dense_108/kernel/Regularizer/Square/ReadVariableOp?
#dense_108/kernel/Regularizer/SquareSquare:dense_108/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_108/kernel/Regularizer/Square?
"dense_108/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_108/kernel/Regularizer/Const?
 dense_108/kernel/Regularizer/SumSum'dense_108/kernel/Regularizer/Square:y:0+dense_108/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/Sum?
"dense_108/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_108/kernel/Regularizer/mul/x?
 dense_108/kernel/Regularizer/mulMul+dense_108/kernel/Regularizer/mul/x:output:0)dense_108/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/mul?
IdentityIdentity*dense_108/StatefulPartitionedCall:output:0"^dense_108/StatefulPartitionedCall3^dense_108/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity)dense_108/ActivityRegularizer/truediv:z:0"^dense_108/StatefulPartitionedCall3^dense_108/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2F
!dense_108/StatefulPartitionedCall!dense_108/StatefulPartitionedCall2h
2dense_108/kernel/Regularizer/Square/ReadVariableOp2dense_108/kernel/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:?????????^
"
_user_specified_name
input_55
?%
?
L__inference_autoencoder_54_layer_call_and_return_conditional_losses_16643606
x)
sequential_108_16643581:^ %
sequential_108_16643583: )
sequential_109_16643587: ^%
sequential_109_16643589:^
identity

identity_1??2dense_108/kernel/Regularizer/Square/ReadVariableOp?2dense_109/kernel/Regularizer/Square/ReadVariableOp?&sequential_108/StatefulPartitionedCall?&sequential_109/StatefulPartitionedCall?
&sequential_108/StatefulPartitionedCallStatefulPartitionedCallxsequential_108_16643581sequential_108_16643583*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_108_layer_call_and_return_conditional_losses_166433162(
&sequential_108/StatefulPartitionedCall?
&sequential_109/StatefulPartitionedCallStatefulPartitionedCall/sequential_108/StatefulPartitionedCall:output:0sequential_109_16643587sequential_109_16643589*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_109_layer_call_and_return_conditional_losses_166434852(
&sequential_109/StatefulPartitionedCall?
2dense_108/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_108_16643581*
_output_shapes

:^ *
dtype024
2dense_108/kernel/Regularizer/Square/ReadVariableOp?
#dense_108/kernel/Regularizer/SquareSquare:dense_108/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_108/kernel/Regularizer/Square?
"dense_108/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_108/kernel/Regularizer/Const?
 dense_108/kernel/Regularizer/SumSum'dense_108/kernel/Regularizer/Square:y:0+dense_108/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/Sum?
"dense_108/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_108/kernel/Regularizer/mul/x?
 dense_108/kernel/Regularizer/mulMul+dense_108/kernel/Regularizer/mul/x:output:0)dense_108/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/mul?
2dense_109/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_109_16643587*
_output_shapes

: ^*
dtype024
2dense_109/kernel/Regularizer/Square/ReadVariableOp?
#dense_109/kernel/Regularizer/SquareSquare:dense_109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_109/kernel/Regularizer/Square?
"dense_109/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_109/kernel/Regularizer/Const?
 dense_109/kernel/Regularizer/SumSum'dense_109/kernel/Regularizer/Square:y:0+dense_109/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/Sum?
"dense_109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_109/kernel/Regularizer/mul/x?
 dense_109/kernel/Regularizer/mulMul+dense_109/kernel/Regularizer/mul/x:output:0)dense_109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/mul?
IdentityIdentity/sequential_109/StatefulPartitionedCall:output:03^dense_108/kernel/Regularizer/Square/ReadVariableOp3^dense_109/kernel/Regularizer/Square/ReadVariableOp'^sequential_108/StatefulPartitionedCall'^sequential_109/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity/sequential_108/StatefulPartitionedCall:output:13^dense_108/kernel/Regularizer/Square/ReadVariableOp3^dense_109/kernel/Regularizer/Square/ReadVariableOp'^sequential_108/StatefulPartitionedCall'^sequential_109/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_108/kernel/Regularizer/Square/ReadVariableOp2dense_108/kernel/Regularizer/Square/ReadVariableOp2h
2dense_109/kernel/Regularizer/Square/ReadVariableOp2dense_109/kernel/Regularizer/Square/ReadVariableOp2P
&sequential_108/StatefulPartitionedCall&sequential_108/StatefulPartitionedCall2P
&sequential_109/StatefulPartitionedCall&sequential_109/StatefulPartitionedCall:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?
?
1__inference_autoencoder_54_layer_call_fn_16643618
input_1
unknown:^ 
	unknown_0: 
	unknown_1: ^
	unknown_2:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????^: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_autoencoder_54_layer_call_and_return_conditional_losses_166436062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????^
!
_user_specified_name	input_1
?
?
1__inference_sequential_108_layer_call_fn_16643943

inputs
unknown:^ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_108_layer_call_and_return_conditional_losses_166433822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?#
?
L__inference_sequential_108_layer_call_and_return_conditional_losses_16643424
input_55$
dense_108_16643403:^  
dense_108_16643405: 
identity

identity_1??!dense_108/StatefulPartitionedCall?2dense_108/kernel/Regularizer/Square/ReadVariableOp?
!dense_108/StatefulPartitionedCallStatefulPartitionedCallinput_55dense_108_16643403dense_108_16643405*
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
GPU 2J 8? *P
fKRI
G__inference_dense_108_layer_call_and_return_conditional_losses_166432942#
!dense_108/StatefulPartitionedCall?
-dense_108/ActivityRegularizer/PartitionedCallPartitionedCall*dense_108/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *<
f7R5
3__inference_dense_108_activity_regularizer_166432702/
-dense_108/ActivityRegularizer/PartitionedCall?
#dense_108/ActivityRegularizer/ShapeShape*dense_108/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2%
#dense_108/ActivityRegularizer/Shape?
1dense_108/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_108/ActivityRegularizer/strided_slice/stack?
3dense_108/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_108/ActivityRegularizer/strided_slice/stack_1?
3dense_108/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_108/ActivityRegularizer/strided_slice/stack_2?
+dense_108/ActivityRegularizer/strided_sliceStridedSlice,dense_108/ActivityRegularizer/Shape:output:0:dense_108/ActivityRegularizer/strided_slice/stack:output:0<dense_108/ActivityRegularizer/strided_slice/stack_1:output:0<dense_108/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_108/ActivityRegularizer/strided_slice?
"dense_108/ActivityRegularizer/CastCast4dense_108/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_108/ActivityRegularizer/Cast?
%dense_108/ActivityRegularizer/truedivRealDiv6dense_108/ActivityRegularizer/PartitionedCall:output:0&dense_108/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_108/ActivityRegularizer/truediv?
2dense_108/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_108_16643403*
_output_shapes

:^ *
dtype024
2dense_108/kernel/Regularizer/Square/ReadVariableOp?
#dense_108/kernel/Regularizer/SquareSquare:dense_108/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_108/kernel/Regularizer/Square?
"dense_108/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_108/kernel/Regularizer/Const?
 dense_108/kernel/Regularizer/SumSum'dense_108/kernel/Regularizer/Square:y:0+dense_108/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/Sum?
"dense_108/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_108/kernel/Regularizer/mul/x?
 dense_108/kernel/Regularizer/mulMul+dense_108/kernel/Regularizer/mul/x:output:0)dense_108/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/mul?
IdentityIdentity*dense_108/StatefulPartitionedCall:output:0"^dense_108/StatefulPartitionedCall3^dense_108/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity)dense_108/ActivityRegularizer/truediv:z:0"^dense_108/StatefulPartitionedCall3^dense_108/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2F
!dense_108/StatefulPartitionedCall!dense_108/StatefulPartitionedCall2h
2dense_108/kernel/Regularizer/Square/ReadVariableOp2dense_108/kernel/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:?????????^
"
_user_specified_name
input_55
?
?
1__inference_sequential_109_layer_call_fn_16644059

inputs
unknown: ^
	unknown_0:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_109_layer_call_and_return_conditional_losses_166434852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

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
?
?
L__inference_sequential_109_layer_call_and_return_conditional_losses_16644145
dense_109_input:
(dense_109_matmul_readvariableop_resource: ^7
)dense_109_biasadd_readvariableop_resource:^
identity?? dense_109/BiasAdd/ReadVariableOp?dense_109/MatMul/ReadVariableOp?2dense_109/kernel/Regularizer/Square/ReadVariableOp?
dense_109/MatMul/ReadVariableOpReadVariableOp(dense_109_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02!
dense_109/MatMul/ReadVariableOp?
dense_109/MatMulMatMuldense_109_input'dense_109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_109/MatMul?
 dense_109/BiasAdd/ReadVariableOpReadVariableOp)dense_109_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02"
 dense_109/BiasAdd/ReadVariableOp?
dense_109/BiasAddBiasAdddense_109/MatMul:product:0(dense_109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_109/BiasAdd
dense_109/SigmoidSigmoiddense_109/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2
dense_109/Sigmoid?
2dense_109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_109_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_109/kernel/Regularizer/Square/ReadVariableOp?
#dense_109/kernel/Regularizer/SquareSquare:dense_109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_109/kernel/Regularizer/Square?
"dense_109/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_109/kernel/Regularizer/Const?
 dense_109/kernel/Regularizer/SumSum'dense_109/kernel/Regularizer/Square:y:0+dense_109/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/Sum?
"dense_109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_109/kernel/Regularizer/mul/x?
 dense_109/kernel/Regularizer/mulMul+dense_109/kernel/Regularizer/mul/x:output:0)dense_109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/mul?
IdentityIdentitydense_109/Sigmoid:y:0!^dense_109/BiasAdd/ReadVariableOp ^dense_109/MatMul/ReadVariableOp3^dense_109/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2D
 dense_109/BiasAdd/ReadVariableOp dense_109/BiasAdd/ReadVariableOp2B
dense_109/MatMul/ReadVariableOpdense_109/MatMul/ReadVariableOp2h
2dense_109/kernel/Regularizer/Square/ReadVariableOp2dense_109/kernel/Regularizer/Square/ReadVariableOp:X T
'
_output_shapes
:????????? 
)
_user_specified_namedense_109_input
?B
?
L__inference_sequential_108_layer_call_and_return_conditional_losses_16644035

inputs:
(dense_108_matmul_readvariableop_resource:^ 7
)dense_108_biasadd_readvariableop_resource: 
identity

identity_1?? dense_108/BiasAdd/ReadVariableOp?dense_108/MatMul/ReadVariableOp?2dense_108/kernel/Regularizer/Square/ReadVariableOp?
dense_108/MatMul/ReadVariableOpReadVariableOp(dense_108_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02!
dense_108/MatMul/ReadVariableOp?
dense_108/MatMulMatMulinputs'dense_108/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_108/MatMul?
 dense_108/BiasAdd/ReadVariableOpReadVariableOp)dense_108_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_108/BiasAdd/ReadVariableOp?
dense_108/BiasAddBiasAdddense_108/MatMul:product:0(dense_108/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_108/BiasAdd
dense_108/SigmoidSigmoiddense_108/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_108/Sigmoid?
4dense_108/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 26
4dense_108/ActivityRegularizer/Mean/reduction_indices?
"dense_108/ActivityRegularizer/MeanMeandense_108/Sigmoid:y:0=dense_108/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2$
"dense_108/ActivityRegularizer/Mean?
'dense_108/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2)
'dense_108/ActivityRegularizer/Maximum/y?
%dense_108/ActivityRegularizer/MaximumMaximum+dense_108/ActivityRegularizer/Mean:output:00dense_108/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2'
%dense_108/ActivityRegularizer/Maximum?
'dense_108/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2)
'dense_108/ActivityRegularizer/truediv/x?
%dense_108/ActivityRegularizer/truedivRealDiv0dense_108/ActivityRegularizer/truediv/x:output:0)dense_108/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2'
%dense_108/ActivityRegularizer/truediv?
!dense_108/ActivityRegularizer/LogLog)dense_108/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2#
!dense_108/ActivityRegularizer/Log?
#dense_108/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_108/ActivityRegularizer/mul/x?
!dense_108/ActivityRegularizer/mulMul,dense_108/ActivityRegularizer/mul/x:output:0%dense_108/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2#
!dense_108/ActivityRegularizer/mul?
#dense_108/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#dense_108/ActivityRegularizer/sub/x?
!dense_108/ActivityRegularizer/subSub,dense_108/ActivityRegularizer/sub/x:output:0)dense_108/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2#
!dense_108/ActivityRegularizer/sub?
)dense_108/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2+
)dense_108/ActivityRegularizer/truediv_1/x?
'dense_108/ActivityRegularizer/truediv_1RealDiv2dense_108/ActivityRegularizer/truediv_1/x:output:0%dense_108/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2)
'dense_108/ActivityRegularizer/truediv_1?
#dense_108/ActivityRegularizer/Log_1Log+dense_108/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2%
#dense_108/ActivityRegularizer/Log_1?
%dense_108/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2'
%dense_108/ActivityRegularizer/mul_1/x?
#dense_108/ActivityRegularizer/mul_1Mul.dense_108/ActivityRegularizer/mul_1/x:output:0'dense_108/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2%
#dense_108/ActivityRegularizer/mul_1?
!dense_108/ActivityRegularizer/addAddV2%dense_108/ActivityRegularizer/mul:z:0'dense_108/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_108/ActivityRegularizer/add?
#dense_108/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#dense_108/ActivityRegularizer/Const?
!dense_108/ActivityRegularizer/SumSum%dense_108/ActivityRegularizer/add:z:0,dense_108/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_108/ActivityRegularizer/Sum?
%dense_108/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%dense_108/ActivityRegularizer/mul_2/x?
#dense_108/ActivityRegularizer/mul_2Mul.dense_108/ActivityRegularizer/mul_2/x:output:0*dense_108/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_108/ActivityRegularizer/mul_2?
#dense_108/ActivityRegularizer/ShapeShapedense_108/Sigmoid:y:0*
T0*
_output_shapes
:2%
#dense_108/ActivityRegularizer/Shape?
1dense_108/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_108/ActivityRegularizer/strided_slice/stack?
3dense_108/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_108/ActivityRegularizer/strided_slice/stack_1?
3dense_108/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_108/ActivityRegularizer/strided_slice/stack_2?
+dense_108/ActivityRegularizer/strided_sliceStridedSlice,dense_108/ActivityRegularizer/Shape:output:0:dense_108/ActivityRegularizer/strided_slice/stack:output:0<dense_108/ActivityRegularizer/strided_slice/stack_1:output:0<dense_108/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_108/ActivityRegularizer/strided_slice?
"dense_108/ActivityRegularizer/CastCast4dense_108/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_108/ActivityRegularizer/Cast?
'dense_108/ActivityRegularizer/truediv_2RealDiv'dense_108/ActivityRegularizer/mul_2:z:0&dense_108/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2)
'dense_108/ActivityRegularizer/truediv_2?
2dense_108/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_108_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_108/kernel/Regularizer/Square/ReadVariableOp?
#dense_108/kernel/Regularizer/SquareSquare:dense_108/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_108/kernel/Regularizer/Square?
"dense_108/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_108/kernel/Regularizer/Const?
 dense_108/kernel/Regularizer/SumSum'dense_108/kernel/Regularizer/Square:y:0+dense_108/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/Sum?
"dense_108/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_108/kernel/Regularizer/mul/x?
 dense_108/kernel/Regularizer/mulMul+dense_108/kernel/Regularizer/mul/x:output:0)dense_108/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/mul?
IdentityIdentitydense_108/Sigmoid:y:0!^dense_108/BiasAdd/ReadVariableOp ^dense_108/MatMul/ReadVariableOp3^dense_108/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity+dense_108/ActivityRegularizer/truediv_2:z:0!^dense_108/BiasAdd/ReadVariableOp ^dense_108/MatMul/ReadVariableOp3^dense_108/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2D
 dense_108/BiasAdd/ReadVariableOp dense_108/BiasAdd/ReadVariableOp2B
dense_108/MatMul/ReadVariableOpdense_108/MatMul/ReadVariableOp2h
2dense_108/kernel/Regularizer/Square/ReadVariableOp2dense_108/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?h
?
L__inference_autoencoder_54_layer_call_and_return_conditional_losses_16643917
xI
7sequential_108_dense_108_matmul_readvariableop_resource:^ F
8sequential_108_dense_108_biasadd_readvariableop_resource: I
7sequential_109_dense_109_matmul_readvariableop_resource: ^F
8sequential_109_dense_109_biasadd_readvariableop_resource:^
identity

identity_1??2dense_108/kernel/Regularizer/Square/ReadVariableOp?2dense_109/kernel/Regularizer/Square/ReadVariableOp?/sequential_108/dense_108/BiasAdd/ReadVariableOp?.sequential_108/dense_108/MatMul/ReadVariableOp?/sequential_109/dense_109/BiasAdd/ReadVariableOp?.sequential_109/dense_109/MatMul/ReadVariableOp?
.sequential_108/dense_108/MatMul/ReadVariableOpReadVariableOp7sequential_108_dense_108_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype020
.sequential_108/dense_108/MatMul/ReadVariableOp?
sequential_108/dense_108/MatMulMatMulx6sequential_108/dense_108/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2!
sequential_108/dense_108/MatMul?
/sequential_108/dense_108/BiasAdd/ReadVariableOpReadVariableOp8sequential_108_dense_108_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential_108/dense_108/BiasAdd/ReadVariableOp?
 sequential_108/dense_108/BiasAddBiasAdd)sequential_108/dense_108/MatMul:product:07sequential_108/dense_108/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2"
 sequential_108/dense_108/BiasAdd?
 sequential_108/dense_108/SigmoidSigmoid)sequential_108/dense_108/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2"
 sequential_108/dense_108/Sigmoid?
Csequential_108/dense_108/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2E
Csequential_108/dense_108/ActivityRegularizer/Mean/reduction_indices?
1sequential_108/dense_108/ActivityRegularizer/MeanMean$sequential_108/dense_108/Sigmoid:y:0Lsequential_108/dense_108/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 23
1sequential_108/dense_108/ActivityRegularizer/Mean?
6sequential_108/dense_108/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.28
6sequential_108/dense_108/ActivityRegularizer/Maximum/y?
4sequential_108/dense_108/ActivityRegularizer/MaximumMaximum:sequential_108/dense_108/ActivityRegularizer/Mean:output:0?sequential_108/dense_108/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 26
4sequential_108/dense_108/ActivityRegularizer/Maximum?
6sequential_108/dense_108/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<28
6sequential_108/dense_108/ActivityRegularizer/truediv/x?
4sequential_108/dense_108/ActivityRegularizer/truedivRealDiv?sequential_108/dense_108/ActivityRegularizer/truediv/x:output:08sequential_108/dense_108/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 26
4sequential_108/dense_108/ActivityRegularizer/truediv?
0sequential_108/dense_108/ActivityRegularizer/LogLog8sequential_108/dense_108/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 22
0sequential_108/dense_108/ActivityRegularizer/Log?
2sequential_108/dense_108/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<24
2sequential_108/dense_108/ActivityRegularizer/mul/x?
0sequential_108/dense_108/ActivityRegularizer/mulMul;sequential_108/dense_108/ActivityRegularizer/mul/x:output:04sequential_108/dense_108/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 22
0sequential_108/dense_108/ActivityRegularizer/mul?
2sequential_108/dense_108/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2sequential_108/dense_108/ActivityRegularizer/sub/x?
0sequential_108/dense_108/ActivityRegularizer/subSub;sequential_108/dense_108/ActivityRegularizer/sub/x:output:08sequential_108/dense_108/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 22
0sequential_108/dense_108/ActivityRegularizer/sub?
8sequential_108/dense_108/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2:
8sequential_108/dense_108/ActivityRegularizer/truediv_1/x?
6sequential_108/dense_108/ActivityRegularizer/truediv_1RealDivAsequential_108/dense_108/ActivityRegularizer/truediv_1/x:output:04sequential_108/dense_108/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 28
6sequential_108/dense_108/ActivityRegularizer/truediv_1?
2sequential_108/dense_108/ActivityRegularizer/Log_1Log:sequential_108/dense_108/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 24
2sequential_108/dense_108/ActivityRegularizer/Log_1?
4sequential_108/dense_108/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?26
4sequential_108/dense_108/ActivityRegularizer/mul_1/x?
2sequential_108/dense_108/ActivityRegularizer/mul_1Mul=sequential_108/dense_108/ActivityRegularizer/mul_1/x:output:06sequential_108/dense_108/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 24
2sequential_108/dense_108/ActivityRegularizer/mul_1?
0sequential_108/dense_108/ActivityRegularizer/addAddV24sequential_108/dense_108/ActivityRegularizer/mul:z:06sequential_108/dense_108/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 22
0sequential_108/dense_108/ActivityRegularizer/add?
2sequential_108/dense_108/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 24
2sequential_108/dense_108/ActivityRegularizer/Const?
0sequential_108/dense_108/ActivityRegularizer/SumSum4sequential_108/dense_108/ActivityRegularizer/add:z:0;sequential_108/dense_108/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 22
0sequential_108/dense_108/ActivityRegularizer/Sum?
4sequential_108/dense_108/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??26
4sequential_108/dense_108/ActivityRegularizer/mul_2/x?
2sequential_108/dense_108/ActivityRegularizer/mul_2Mul=sequential_108/dense_108/ActivityRegularizer/mul_2/x:output:09sequential_108/dense_108/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 24
2sequential_108/dense_108/ActivityRegularizer/mul_2?
2sequential_108/dense_108/ActivityRegularizer/ShapeShape$sequential_108/dense_108/Sigmoid:y:0*
T0*
_output_shapes
:24
2sequential_108/dense_108/ActivityRegularizer/Shape?
@sequential_108/dense_108/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential_108/dense_108/ActivityRegularizer/strided_slice/stack?
Bsequential_108/dense_108/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_108/dense_108/ActivityRegularizer/strided_slice/stack_1?
Bsequential_108/dense_108/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_108/dense_108/ActivityRegularizer/strided_slice/stack_2?
:sequential_108/dense_108/ActivityRegularizer/strided_sliceStridedSlice;sequential_108/dense_108/ActivityRegularizer/Shape:output:0Isequential_108/dense_108/ActivityRegularizer/strided_slice/stack:output:0Ksequential_108/dense_108/ActivityRegularizer/strided_slice/stack_1:output:0Ksequential_108/dense_108/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_108/dense_108/ActivityRegularizer/strided_slice?
1sequential_108/dense_108/ActivityRegularizer/CastCastCsequential_108/dense_108/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 23
1sequential_108/dense_108/ActivityRegularizer/Cast?
6sequential_108/dense_108/ActivityRegularizer/truediv_2RealDiv6sequential_108/dense_108/ActivityRegularizer/mul_2:z:05sequential_108/dense_108/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 28
6sequential_108/dense_108/ActivityRegularizer/truediv_2?
.sequential_109/dense_109/MatMul/ReadVariableOpReadVariableOp7sequential_109_dense_109_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype020
.sequential_109/dense_109/MatMul/ReadVariableOp?
sequential_109/dense_109/MatMulMatMul$sequential_108/dense_108/Sigmoid:y:06sequential_109/dense_109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2!
sequential_109/dense_109/MatMul?
/sequential_109/dense_109/BiasAdd/ReadVariableOpReadVariableOp8sequential_109_dense_109_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype021
/sequential_109/dense_109/BiasAdd/ReadVariableOp?
 sequential_109/dense_109/BiasAddBiasAdd)sequential_109/dense_109/MatMul:product:07sequential_109/dense_109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2"
 sequential_109/dense_109/BiasAdd?
 sequential_109/dense_109/SigmoidSigmoid)sequential_109/dense_109/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2"
 sequential_109/dense_109/Sigmoid?
2dense_108/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7sequential_108_dense_108_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_108/kernel/Regularizer/Square/ReadVariableOp?
#dense_108/kernel/Regularizer/SquareSquare:dense_108/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_108/kernel/Regularizer/Square?
"dense_108/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_108/kernel/Regularizer/Const?
 dense_108/kernel/Regularizer/SumSum'dense_108/kernel/Regularizer/Square:y:0+dense_108/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/Sum?
"dense_108/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_108/kernel/Regularizer/mul/x?
 dense_108/kernel/Regularizer/mulMul+dense_108/kernel/Regularizer/mul/x:output:0)dense_108/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/mul?
2dense_109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7sequential_109_dense_109_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_109/kernel/Regularizer/Square/ReadVariableOp?
#dense_109/kernel/Regularizer/SquareSquare:dense_109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_109/kernel/Regularizer/Square?
"dense_109/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_109/kernel/Regularizer/Const?
 dense_109/kernel/Regularizer/SumSum'dense_109/kernel/Regularizer/Square:y:0+dense_109/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/Sum?
"dense_109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_109/kernel/Regularizer/mul/x?
 dense_109/kernel/Regularizer/mulMul+dense_109/kernel/Regularizer/mul/x:output:0)dense_109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/mul?
IdentityIdentity$sequential_109/dense_109/Sigmoid:y:03^dense_108/kernel/Regularizer/Square/ReadVariableOp3^dense_109/kernel/Regularizer/Square/ReadVariableOp0^sequential_108/dense_108/BiasAdd/ReadVariableOp/^sequential_108/dense_108/MatMul/ReadVariableOp0^sequential_109/dense_109/BiasAdd/ReadVariableOp/^sequential_109/dense_109/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity:sequential_108/dense_108/ActivityRegularizer/truediv_2:z:03^dense_108/kernel/Regularizer/Square/ReadVariableOp3^dense_109/kernel/Regularizer/Square/ReadVariableOp0^sequential_108/dense_108/BiasAdd/ReadVariableOp/^sequential_108/dense_108/MatMul/ReadVariableOp0^sequential_109/dense_109/BiasAdd/ReadVariableOp/^sequential_109/dense_109/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_108/kernel/Regularizer/Square/ReadVariableOp2dense_108/kernel/Regularizer/Square/ReadVariableOp2h
2dense_109/kernel/Regularizer/Square/ReadVariableOp2dense_109/kernel/Regularizer/Square/ReadVariableOp2b
/sequential_108/dense_108/BiasAdd/ReadVariableOp/sequential_108/dense_108/BiasAdd/ReadVariableOp2`
.sequential_108/dense_108/MatMul/ReadVariableOp.sequential_108/dense_108/MatMul/ReadVariableOp2b
/sequential_109/dense_109/BiasAdd/ReadVariableOp/sequential_109/dense_109/BiasAdd/ReadVariableOp2`
.sequential_109/dense_109/MatMul/ReadVariableOp.sequential_109/dense_109/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?B
?
L__inference_sequential_108_layer_call_and_return_conditional_losses_16643989

inputs:
(dense_108_matmul_readvariableop_resource:^ 7
)dense_108_biasadd_readvariableop_resource: 
identity

identity_1?? dense_108/BiasAdd/ReadVariableOp?dense_108/MatMul/ReadVariableOp?2dense_108/kernel/Regularizer/Square/ReadVariableOp?
dense_108/MatMul/ReadVariableOpReadVariableOp(dense_108_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02!
dense_108/MatMul/ReadVariableOp?
dense_108/MatMulMatMulinputs'dense_108/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_108/MatMul?
 dense_108/BiasAdd/ReadVariableOpReadVariableOp)dense_108_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_108/BiasAdd/ReadVariableOp?
dense_108/BiasAddBiasAdddense_108/MatMul:product:0(dense_108/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_108/BiasAdd
dense_108/SigmoidSigmoiddense_108/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_108/Sigmoid?
4dense_108/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 26
4dense_108/ActivityRegularizer/Mean/reduction_indices?
"dense_108/ActivityRegularizer/MeanMeandense_108/Sigmoid:y:0=dense_108/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2$
"dense_108/ActivityRegularizer/Mean?
'dense_108/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2)
'dense_108/ActivityRegularizer/Maximum/y?
%dense_108/ActivityRegularizer/MaximumMaximum+dense_108/ActivityRegularizer/Mean:output:00dense_108/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2'
%dense_108/ActivityRegularizer/Maximum?
'dense_108/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2)
'dense_108/ActivityRegularizer/truediv/x?
%dense_108/ActivityRegularizer/truedivRealDiv0dense_108/ActivityRegularizer/truediv/x:output:0)dense_108/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2'
%dense_108/ActivityRegularizer/truediv?
!dense_108/ActivityRegularizer/LogLog)dense_108/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2#
!dense_108/ActivityRegularizer/Log?
#dense_108/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_108/ActivityRegularizer/mul/x?
!dense_108/ActivityRegularizer/mulMul,dense_108/ActivityRegularizer/mul/x:output:0%dense_108/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2#
!dense_108/ActivityRegularizer/mul?
#dense_108/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#dense_108/ActivityRegularizer/sub/x?
!dense_108/ActivityRegularizer/subSub,dense_108/ActivityRegularizer/sub/x:output:0)dense_108/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2#
!dense_108/ActivityRegularizer/sub?
)dense_108/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2+
)dense_108/ActivityRegularizer/truediv_1/x?
'dense_108/ActivityRegularizer/truediv_1RealDiv2dense_108/ActivityRegularizer/truediv_1/x:output:0%dense_108/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2)
'dense_108/ActivityRegularizer/truediv_1?
#dense_108/ActivityRegularizer/Log_1Log+dense_108/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2%
#dense_108/ActivityRegularizer/Log_1?
%dense_108/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2'
%dense_108/ActivityRegularizer/mul_1/x?
#dense_108/ActivityRegularizer/mul_1Mul.dense_108/ActivityRegularizer/mul_1/x:output:0'dense_108/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2%
#dense_108/ActivityRegularizer/mul_1?
!dense_108/ActivityRegularizer/addAddV2%dense_108/ActivityRegularizer/mul:z:0'dense_108/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_108/ActivityRegularizer/add?
#dense_108/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#dense_108/ActivityRegularizer/Const?
!dense_108/ActivityRegularizer/SumSum%dense_108/ActivityRegularizer/add:z:0,dense_108/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_108/ActivityRegularizer/Sum?
%dense_108/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%dense_108/ActivityRegularizer/mul_2/x?
#dense_108/ActivityRegularizer/mul_2Mul.dense_108/ActivityRegularizer/mul_2/x:output:0*dense_108/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_108/ActivityRegularizer/mul_2?
#dense_108/ActivityRegularizer/ShapeShapedense_108/Sigmoid:y:0*
T0*
_output_shapes
:2%
#dense_108/ActivityRegularizer/Shape?
1dense_108/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_108/ActivityRegularizer/strided_slice/stack?
3dense_108/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_108/ActivityRegularizer/strided_slice/stack_1?
3dense_108/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_108/ActivityRegularizer/strided_slice/stack_2?
+dense_108/ActivityRegularizer/strided_sliceStridedSlice,dense_108/ActivityRegularizer/Shape:output:0:dense_108/ActivityRegularizer/strided_slice/stack:output:0<dense_108/ActivityRegularizer/strided_slice/stack_1:output:0<dense_108/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_108/ActivityRegularizer/strided_slice?
"dense_108/ActivityRegularizer/CastCast4dense_108/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_108/ActivityRegularizer/Cast?
'dense_108/ActivityRegularizer/truediv_2RealDiv'dense_108/ActivityRegularizer/mul_2:z:0&dense_108/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2)
'dense_108/ActivityRegularizer/truediv_2?
2dense_108/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_108_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_108/kernel/Regularizer/Square/ReadVariableOp?
#dense_108/kernel/Regularizer/SquareSquare:dense_108/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_108/kernel/Regularizer/Square?
"dense_108/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_108/kernel/Regularizer/Const?
 dense_108/kernel/Regularizer/SumSum'dense_108/kernel/Regularizer/Square:y:0+dense_108/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/Sum?
"dense_108/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_108/kernel/Regularizer/mul/x?
 dense_108/kernel/Regularizer/mulMul+dense_108/kernel/Regularizer/mul/x:output:0)dense_108/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/mul?
IdentityIdentitydense_108/Sigmoid:y:0!^dense_108/BiasAdd/ReadVariableOp ^dense_108/MatMul/ReadVariableOp3^dense_108/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity+dense_108/ActivityRegularizer/truediv_2:z:0!^dense_108/BiasAdd/ReadVariableOp ^dense_108/MatMul/ReadVariableOp3^dense_108/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2D
 dense_108/BiasAdd/ReadVariableOp dense_108/BiasAdd/ReadVariableOp2B
dense_108/MatMul/ReadVariableOpdense_108/MatMul/ReadVariableOp2h
2dense_108/kernel/Regularizer/Square/ReadVariableOp2dense_108/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?%
?
L__inference_autoencoder_54_layer_call_and_return_conditional_losses_16643662
x)
sequential_108_16643637:^ %
sequential_108_16643639: )
sequential_109_16643643: ^%
sequential_109_16643645:^
identity

identity_1??2dense_108/kernel/Regularizer/Square/ReadVariableOp?2dense_109/kernel/Regularizer/Square/ReadVariableOp?&sequential_108/StatefulPartitionedCall?&sequential_109/StatefulPartitionedCall?
&sequential_108/StatefulPartitionedCallStatefulPartitionedCallxsequential_108_16643637sequential_108_16643639*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:????????? : *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_108_layer_call_and_return_conditional_losses_166433822(
&sequential_108/StatefulPartitionedCall?
&sequential_109/StatefulPartitionedCallStatefulPartitionedCall/sequential_108/StatefulPartitionedCall:output:0sequential_109_16643643sequential_109_16643645*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????^*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_sequential_109_layer_call_and_return_conditional_losses_166435282(
&sequential_109/StatefulPartitionedCall?
2dense_108/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_108_16643637*
_output_shapes

:^ *
dtype024
2dense_108/kernel/Regularizer/Square/ReadVariableOp?
#dense_108/kernel/Regularizer/SquareSquare:dense_108/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_108/kernel/Regularizer/Square?
"dense_108/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_108/kernel/Regularizer/Const?
 dense_108/kernel/Regularizer/SumSum'dense_108/kernel/Regularizer/Square:y:0+dense_108/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/Sum?
"dense_108/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_108/kernel/Regularizer/mul/x?
 dense_108/kernel/Regularizer/mulMul+dense_108/kernel/Regularizer/mul/x:output:0)dense_108/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/mul?
2dense_109/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_109_16643643*
_output_shapes

: ^*
dtype024
2dense_109/kernel/Regularizer/Square/ReadVariableOp?
#dense_109/kernel/Regularizer/SquareSquare:dense_109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_109/kernel/Regularizer/Square?
"dense_109/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_109/kernel/Regularizer/Const?
 dense_109/kernel/Regularizer/SumSum'dense_109/kernel/Regularizer/Square:y:0+dense_109/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/Sum?
"dense_109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_109/kernel/Regularizer/mul/x?
 dense_109/kernel/Regularizer/mulMul+dense_109/kernel/Regularizer/mul/x:output:0)dense_109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/mul?
IdentityIdentity/sequential_109/StatefulPartitionedCall:output:03^dense_108/kernel/Regularizer/Square/ReadVariableOp3^dense_109/kernel/Regularizer/Square/ReadVariableOp'^sequential_108/StatefulPartitionedCall'^sequential_109/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity/sequential_108/StatefulPartitionedCall:output:13^dense_108/kernel/Regularizer/Square/ReadVariableOp3^dense_109/kernel/Regularizer/Square/ReadVariableOp'^sequential_108/StatefulPartitionedCall'^sequential_109/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_108/kernel/Regularizer/Square/ReadVariableOp2dense_108/kernel/Regularizer/Square/ReadVariableOp2h
2dense_109/kernel/Regularizer/Square/ReadVariableOp2dense_109/kernel/Regularizer/Square/ReadVariableOp2P
&sequential_108/StatefulPartitionedCall&sequential_108/StatefulPartitionedCall2P
&sequential_109/StatefulPartitionedCall&sequential_109/StatefulPartitionedCall:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?
?
L__inference_sequential_109_layer_call_and_return_conditional_losses_16644094

inputs:
(dense_109_matmul_readvariableop_resource: ^7
)dense_109_biasadd_readvariableop_resource:^
identity?? dense_109/BiasAdd/ReadVariableOp?dense_109/MatMul/ReadVariableOp?2dense_109/kernel/Regularizer/Square/ReadVariableOp?
dense_109/MatMul/ReadVariableOpReadVariableOp(dense_109_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02!
dense_109/MatMul/ReadVariableOp?
dense_109/MatMulMatMulinputs'dense_109/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_109/MatMul?
 dense_109/BiasAdd/ReadVariableOpReadVariableOp)dense_109_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02"
 dense_109/BiasAdd/ReadVariableOp?
dense_109/BiasAddBiasAdddense_109/MatMul:product:0(dense_109/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_109/BiasAdd
dense_109/SigmoidSigmoiddense_109/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2
dense_109/Sigmoid?
2dense_109/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_109_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_109/kernel/Regularizer/Square/ReadVariableOp?
#dense_109/kernel/Regularizer/SquareSquare:dense_109/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_109/kernel/Regularizer/Square?
"dense_109/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_109/kernel/Regularizer/Const?
 dense_109/kernel/Regularizer/SumSum'dense_109/kernel/Regularizer/Square:y:0+dense_109/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/Sum?
"dense_109/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_109/kernel/Regularizer/mul/x?
 dense_109/kernel/Regularizer/mulMul+dense_109/kernel/Regularizer/mul/x:output:0)dense_109/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_109/kernel/Regularizer/mul?
IdentityIdentitydense_109/Sigmoid:y:0!^dense_109/BiasAdd/ReadVariableOp ^dense_109/MatMul/ReadVariableOp3^dense_109/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2D
 dense_109/BiasAdd/ReadVariableOp dense_109/BiasAdd/ReadVariableOp2B
dense_109/MatMul/ReadVariableOpdense_109/MatMul/ReadVariableOp2h
2dense_109/kernel/Regularizer/Square/ReadVariableOp2dense_109/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_16644182M
;dense_108_kernel_regularizer_square_readvariableop_resource:^ 
identity??2dense_108/kernel/Regularizer/Square/ReadVariableOp?
2dense_108/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;dense_108_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_108/kernel/Regularizer/Square/ReadVariableOp?
#dense_108/kernel/Regularizer/SquareSquare:dense_108/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_108/kernel/Regularizer/Square?
"dense_108/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_108/kernel/Regularizer/Const?
 dense_108/kernel/Regularizer/SumSum'dense_108/kernel/Regularizer/Square:y:0+dense_108/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/Sum?
"dense_108/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_108/kernel/Regularizer/mul/x?
 dense_108/kernel/Regularizer/mulMul+dense_108/kernel/Regularizer/mul/x:output:0)dense_108/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_108/kernel/Regularizer/mul?
IdentityIdentity$dense_108/kernel/Regularizer/mul:z:03^dense_108/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_108/kernel/Regularizer/Square/ReadVariableOp2dense_108/kernel/Regularizer/Square/ReadVariableOp
?
?
,__inference_dense_108_layer_call_fn_16644160

inputs
unknown:^ 
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
GPU 2J 8? *P
fKRI
G__inference_dense_108_layer_call_and_return_conditional_losses_166432942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs"?L
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
serving_default_input_1:0?????????^<
output_10
StatefulPartitionedCall:0?????????^tensorflow/serving/predict:??
?
history
encoder
decoder
trainable_variables
	variables
regularization_losses
	keras_api

signatures
8__call__
9_default_save_signature
*:&call_and_return_all_conditional_losses"?
_tf_keras_model?{"name": "autoencoder_54", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Autoencoder", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 94]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Autoencoder"}}
 "
trackable_dict_wrapper
?
	layer_with_weights-0
	layer-0

trainable_variables
	variables
regularization_losses
	keras_api
;__call__
*<&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"name": "sequential_108", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_108", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 94]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_55"}}, {"class_name": "Dense", "config": {"name": "dense_108", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}}, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 94}}, "shared_object_id": 6}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 94]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 94]}, "float32", "input_55"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_108", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 94]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_55"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_108", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}]}}}
?
layer_with_weights-0
layer-0
trainable_variables
	variables
regularization_losses
	keras_api
=__call__
*>&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"name": "sequential_109", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_109", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_109_input"}}, {"class_name": "Dense", "config": {"name": "dense_109", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [94, 32]}, "float32", "dense_109_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_109", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_109_input"}, "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "dense_109", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8}]}}}
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
layer_regularization_losses
non_trainable_variables
metrics
trainable_variables

layers
	variables
layer_metrics
regularization_losses
8__call__
9_default_save_signature
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
,
?serving_default"
signature_map
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
@__call__
*A&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "dense_108", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_108", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 94}}, "shared_object_id": 6}, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 94]}}
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
 layer_regularization_losses
!non_trainable_variables
"metrics

trainable_variables

#layers
	variables
$layer_metrics
regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
?	

kernel
bias
%trainable_variables
&	variables
'regularization_losses
(	keras_api
C__call__
*D&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_109", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_109", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [94, 32]}}
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
)layer_regularization_losses
*non_trainable_variables
+metrics
trainable_variables

,layers
	variables
-layer_metrics
regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
": ^ 2dense_108/kernel
: 2dense_108/bias
":  ^2dense_109/kernel
:^2dense_109/bias
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
 "
trackable_dict_wrapper
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
.layer_regularization_losses
/metrics
0non_trainable_variables
trainable_variables

1layers
	variables
2layer_metrics
regularization_losses
@__call__
Factivity_regularizer_fn
*A&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
	0"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
3layer_regularization_losses
4metrics
5non_trainable_variables
%trainable_variables

6layers
&	variables
7layer_metrics
'regularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
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
'
B0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
E0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
1__inference_autoencoder_54_layer_call_fn_16643618
1__inference_autoencoder_54_layer_call_fn_16643785
1__inference_autoencoder_54_layer_call_fn_16643799
1__inference_autoencoder_54_layer_call_fn_16643688?
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
#__inference__wrapped_model_16643241?
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
input_1?????????^
?2?
L__inference_autoencoder_54_layer_call_and_return_conditional_losses_16643858
L__inference_autoencoder_54_layer_call_and_return_conditional_losses_16643917
L__inference_autoencoder_54_layer_call_and_return_conditional_losses_16643716
L__inference_autoencoder_54_layer_call_and_return_conditional_losses_16643744?
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
1__inference_sequential_108_layer_call_fn_16643324
1__inference_sequential_108_layer_call_fn_16643933
1__inference_sequential_108_layer_call_fn_16643943
1__inference_sequential_108_layer_call_fn_16643400?
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
L__inference_sequential_108_layer_call_and_return_conditional_losses_16643989
L__inference_sequential_108_layer_call_and_return_conditional_losses_16644035
L__inference_sequential_108_layer_call_and_return_conditional_losses_16643424
L__inference_sequential_108_layer_call_and_return_conditional_losses_16643448?
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
1__inference_sequential_109_layer_call_fn_16644050
1__inference_sequential_109_layer_call_fn_16644059
1__inference_sequential_109_layer_call_fn_16644068
1__inference_sequential_109_layer_call_fn_16644077?
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
L__inference_sequential_109_layer_call_and_return_conditional_losses_16644094
L__inference_sequential_109_layer_call_and_return_conditional_losses_16644111
L__inference_sequential_109_layer_call_and_return_conditional_losses_16644128
L__inference_sequential_109_layer_call_and_return_conditional_losses_16644145?
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
&__inference_signature_wrapper_16643771input_1"?
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
,__inference_dense_108_layer_call_fn_16644160?
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
K__inference_dense_108_layer_call_and_return_all_conditional_losses_16644171?
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
__inference_loss_fn_0_16644182?
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
,__inference_dense_109_layer_call_fn_16644197?
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
G__inference_dense_109_layer_call_and_return_conditional_losses_16644214?
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
__inference_loss_fn_1_16644225?
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
3__inference_dense_108_activity_regularizer_16643270?
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
G__inference_dense_108_layer_call_and_return_conditional_losses_16644242?
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
#__inference__wrapped_model_16643241m0?-
&?#
!?
input_1?????????^
? "3?0
.
output_1"?
output_1?????????^?
L__inference_autoencoder_54_layer_call_and_return_conditional_losses_16643716q4?1
*?'
!?
input_1?????????^
p 
? "3?0
?
0?????????^
?
?	
1/0 ?
L__inference_autoencoder_54_layer_call_and_return_conditional_losses_16643744q4?1
*?'
!?
input_1?????????^
p
? "3?0
?
0?????????^
?
?	
1/0 ?
L__inference_autoencoder_54_layer_call_and_return_conditional_losses_16643858k.?+
$?!
?
X?????????^
p 
? "3?0
?
0?????????^
?
?	
1/0 ?
L__inference_autoencoder_54_layer_call_and_return_conditional_losses_16643917k.?+
$?!
?
X?????????^
p
? "3?0
?
0?????????^
?
?	
1/0 ?
1__inference_autoencoder_54_layer_call_fn_16643618V4?1
*?'
!?
input_1?????????^
p 
? "??????????^?
1__inference_autoencoder_54_layer_call_fn_16643688V4?1
*?'
!?
input_1?????????^
p
? "??????????^?
1__inference_autoencoder_54_layer_call_fn_16643785P.?+
$?!
?
X?????????^
p 
? "??????????^?
1__inference_autoencoder_54_layer_call_fn_16643799P.?+
$?!
?
X?????????^
p
? "??????????^f
3__inference_dense_108_activity_regularizer_16643270/$?!
?
?

activation
? "? ?
K__inference_dense_108_layer_call_and_return_all_conditional_losses_16644171j/?,
%?"
 ?
inputs?????????^
? "3?0
?
0????????? 
?
?	
1/0 ?
G__inference_dense_108_layer_call_and_return_conditional_losses_16644242\/?,
%?"
 ?
inputs?????????^
? "%?"
?
0????????? 
? 
,__inference_dense_108_layer_call_fn_16644160O/?,
%?"
 ?
inputs?????????^
? "?????????? ?
G__inference_dense_109_layer_call_and_return_conditional_losses_16644214\/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????^
? 
,__inference_dense_109_layer_call_fn_16644197O/?,
%?"
 ?
inputs????????? 
? "??????????^=
__inference_loss_fn_0_16644182?

? 
? "? =
__inference_loss_fn_1_16644225?

? 
? "? ?
L__inference_sequential_108_layer_call_and_return_conditional_losses_16643424t9?6
/?,
"?
input_55?????????^
p 

 
? "3?0
?
0????????? 
?
?	
1/0 ?
L__inference_sequential_108_layer_call_and_return_conditional_losses_16643448t9?6
/?,
"?
input_55?????????^
p

 
? "3?0
?
0????????? 
?
?	
1/0 ?
L__inference_sequential_108_layer_call_and_return_conditional_losses_16643989r7?4
-?*
 ?
inputs?????????^
p 

 
? "3?0
?
0????????? 
?
?	
1/0 ?
L__inference_sequential_108_layer_call_and_return_conditional_losses_16644035r7?4
-?*
 ?
inputs?????????^
p

 
? "3?0
?
0????????? 
?
?	
1/0 ?
1__inference_sequential_108_layer_call_fn_16643324Y9?6
/?,
"?
input_55?????????^
p 

 
? "?????????? ?
1__inference_sequential_108_layer_call_fn_16643400Y9?6
/?,
"?
input_55?????????^
p

 
? "?????????? ?
1__inference_sequential_108_layer_call_fn_16643933W7?4
-?*
 ?
inputs?????????^
p 

 
? "?????????? ?
1__inference_sequential_108_layer_call_fn_16643943W7?4
-?*
 ?
inputs?????????^
p

 
? "?????????? ?
L__inference_sequential_109_layer_call_and_return_conditional_losses_16644094d7?4
-?*
 ?
inputs????????? 
p 

 
? "%?"
?
0?????????^
? ?
L__inference_sequential_109_layer_call_and_return_conditional_losses_16644111d7?4
-?*
 ?
inputs????????? 
p

 
? "%?"
?
0?????????^
? ?
L__inference_sequential_109_layer_call_and_return_conditional_losses_16644128m@?=
6?3
)?&
dense_109_input????????? 
p 

 
? "%?"
?
0?????????^
? ?
L__inference_sequential_109_layer_call_and_return_conditional_losses_16644145m@?=
6?3
)?&
dense_109_input????????? 
p

 
? "%?"
?
0?????????^
? ?
1__inference_sequential_109_layer_call_fn_16644050`@?=
6?3
)?&
dense_109_input????????? 
p 

 
? "??????????^?
1__inference_sequential_109_layer_call_fn_16644059W7?4
-?*
 ?
inputs????????? 
p 

 
? "??????????^?
1__inference_sequential_109_layer_call_fn_16644068W7?4
-?*
 ?
inputs????????? 
p

 
? "??????????^?
1__inference_sequential_109_layer_call_fn_16644077`@?=
6?3
)?&
dense_109_input????????? 
p

 
? "??????????^?
&__inference_signature_wrapper_16643771x;?8
? 
1?.
,
input_1!?
input_1?????????^"3?0
.
output_1"?
output_1?????????^