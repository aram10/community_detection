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
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
??*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:?*
dtype0
z
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_3/kernel
s
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel* 
_output_shapes
:
??*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:?*
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
layer_regularization_losses
	variables
non_trainable_variables
layer_metrics
trainable_variables
regularization_losses
metrics

layers
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
 layer_regularization_losses

	variables
!non_trainable_variables
"layer_metrics
trainable_variables
regularization_losses
#metrics

$layers
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
)layer_regularization_losses
	variables
*non_trainable_variables
+layer_metrics
trainable_variables
regularization_losses
,metrics

-layers
JH
VARIABLE_VALUEdense_2/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_2/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_3/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_3/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
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
?
.layer_regularization_losses
	variables
/non_trainable_variables
0layer_metrics
trainable_variables
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
?
3layer_regularization_losses
%	variables
4non_trainable_variables
5layer_metrics
&trainable_variables
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
|
serving_default_input_1Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_19224159
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpConst*
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
!__inference__traced_save_19224669
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
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
$__inference__traced_restore_19224691??
?$
?
K__inference_autoencoder_1_layer_call_and_return_conditional_losses_19224050
x)
sequential_2_19224025:
??$
sequential_2_19224027:	?)
sequential_3_19224031:
??$
sequential_3_19224033:	?
identity

identity_1??0dense_2/kernel/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?$sequential_2/StatefulPartitionedCall?$sequential_3/StatefulPartitionedCall?
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallxsequential_2_19224025sequential_2_19224027*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_192237702&
$sequential_2/StatefulPartitionedCall?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_19224031sequential_3_19224033*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_192239162&
$sequential_3/StatefulPartitionedCall?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_2_19224025* 
_output_shapes
:
??*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_3_19224031* 
_output_shapes
:
??*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:01^dense_2/kernel/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity-sequential_2/StatefulPartitionedCall:output:11^dense_2/kernel/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_nameX
?
?
/__inference_sequential_3_layer_call_fn_19224451

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_192238732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?A
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_19224380

inputs:
&dense_2_matmul_readvariableop_resource:
??6
'dense_2_biasadd_readvariableop_resource:	?
identity

identity_1??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddz
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_2/Sigmoid?
#dense_2/ActivityRegularizer/SigmoidSigmoiddense_2/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2%
#dense_2/ActivityRegularizer/Sigmoid?
2dense_2/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 24
2dense_2/ActivityRegularizer/Mean/reduction_indices?
 dense_2/ActivityRegularizer/MeanMean'dense_2/ActivityRegularizer/Sigmoid:y:0;dense_2/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes	
:?2"
 dense_2/ActivityRegularizer/Mean?
%dense_2/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2'
%dense_2/ActivityRegularizer/Maximum/y?
#dense_2/ActivityRegularizer/MaximumMaximum)dense_2/ActivityRegularizer/Mean:output:0.dense_2/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes	
:?2%
#dense_2/ActivityRegularizer/Maximum?
%dense_2/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2'
%dense_2/ActivityRegularizer/truediv/x?
#dense_2/ActivityRegularizer/truedivRealDiv.dense_2/ActivityRegularizer/truediv/x:output:0'dense_2/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2%
#dense_2/ActivityRegularizer/truediv?
dense_2/ActivityRegularizer/LogLog'dense_2/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes	
:?2!
dense_2/ActivityRegularizer/Log?
!dense_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_2/ActivityRegularizer/mul/x?
dense_2/ActivityRegularizer/mulMul*dense_2/ActivityRegularizer/mul/x:output:0#dense_2/ActivityRegularizer/Log:y:0*
T0*
_output_shapes	
:?2!
dense_2/ActivityRegularizer/mul?
!dense_2/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!dense_2/ActivityRegularizer/sub/x?
dense_2/ActivityRegularizer/subSub*dense_2/ActivityRegularizer/sub/x:output:0'dense_2/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2!
dense_2/ActivityRegularizer/sub?
'dense_2/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2)
'dense_2/ActivityRegularizer/truediv_1/x?
%dense_2/ActivityRegularizer/truediv_1RealDiv0dense_2/ActivityRegularizer/truediv_1/x:output:0#dense_2/ActivityRegularizer/sub:z:0*
T0*
_output_shapes	
:?2'
%dense_2/ActivityRegularizer/truediv_1?
!dense_2/ActivityRegularizer/Log_1Log)dense_2/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes	
:?2#
!dense_2/ActivityRegularizer/Log_1?
#dense_2/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2%
#dense_2/ActivityRegularizer/mul_1/x?
!dense_2/ActivityRegularizer/mul_1Mul,dense_2/ActivityRegularizer/mul_1/x:output:0%dense_2/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes	
:?2#
!dense_2/ActivityRegularizer/mul_1?
dense_2/ActivityRegularizer/addAddV2#dense_2/ActivityRegularizer/mul:z:0%dense_2/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes	
:?2!
dense_2/ActivityRegularizer/add?
!dense_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_2/ActivityRegularizer/Const?
dense_2/ActivityRegularizer/SumSum#dense_2/ActivityRegularizer/add:z:0*dense_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_2/ActivityRegularizer/Sum?
#dense_2/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#dense_2/ActivityRegularizer/mul_2/x?
!dense_2/ActivityRegularizer/mul_2Mul,dense_2/ActivityRegularizer/mul_2/x:output:0(dense_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_2/ActivityRegularizer/mul_2?
!dense_2/ActivityRegularizer/ShapeShapedense_2/Sigmoid:y:0*
T0*
_output_shapes
:2#
!dense_2/ActivityRegularizer/Shape?
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_2/ActivityRegularizer/strided_slice/stack?
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_1?
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_2?
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_2/ActivityRegularizer/strided_slice?
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_2/ActivityRegularizer/Cast?
%dense_2/ActivityRegularizer/truediv_2RealDiv%dense_2/ActivityRegularizer/mul_2:z:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_2/ActivityRegularizer/truediv_2?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
IdentityIdentitydense_2/Sigmoid:y:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity)dense_2/ActivityRegularizer/truediv_2:z:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_3_layer_call_fn_19224469
dense_3_input
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_3_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_192239162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_namedense_3_input
?
?
__inference_loss_fn_1_19224617M
9dense_3_kernel_regularizer_square_readvariableop_resource:
??
identity??0dense_3/kernel/Regularizer/Square/ReadVariableOp?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_3_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
IdentityIdentity"dense_3/kernel/Regularizer/mul:z:01^dense_3/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp
?
?
&__inference_signature_wrapper_19224159
input_1
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_192236282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
0__inference_autoencoder_1_layer_call_fn_19224076
input_1
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_autoencoder_1_layer_call_and_return_conditional_losses_192240502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
/__inference_sequential_3_layer_call_fn_19224442
dense_3_input
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_3_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_192238732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_namedense_3_input
?
?
E__inference_dense_2_layer_call_and_return_conditional_losses_19223682

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoid?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?"
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_19223704

inputs$
dense_2_19223683:
??
dense_2_19223685:	?
identity

identity_1??dense_2/StatefulPartitionedCall?0dense_2/kernel/Regularizer/Square/ReadVariableOp?
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_19223683dense_2_19223685*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_192236822!
dense_2/StatefulPartitionedCall?
+dense_2/ActivityRegularizer/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *:
f5R3
1__inference_dense_2_activity_regularizer_192236582-
+dense_2/ActivityRegularizer/PartitionedCall?
!dense_2/ActivityRegularizer/ShapeShape(dense_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_2/ActivityRegularizer/Shape?
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_2/ActivityRegularizer/strided_slice/stack?
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_1?
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_2?
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_2/ActivityRegularizer/strided_slice?
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_2/ActivityRegularizer/Cast?
#dense_2/ActivityRegularizer/truedivRealDiv4dense_2/ActivityRegularizer/PartitionedCall:output:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_2/ActivityRegularizer/truediv?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_19223683* 
_output_shapes
:
??*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity'dense_2/ActivityRegularizer/truediv:z:0 ^dense_2/StatefulPartitionedCall1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_3_layer_call_and_return_conditional_losses_19223873

inputs$
dense_3_19223861:
??
dense_3_19223863:	?
identity??dense_3/StatefulPartitionedCall?0dense_3/kernel/Regularizer/Square/ReadVariableOp?
dense_3/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_19223861dense_3_19223863*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_192238602!
dense_3/StatefulPartitionedCall?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_19223861* 
_output_shapes
:
??*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall1^dense_3/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_3_layer_call_and_return_conditional_losses_19224503

inputs:
&dense_3_matmul_readvariableop_resource:
??6
'dense_3_biasadd_readvariableop_resource:	?
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/BiasAddz
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_3/Sigmoid?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
IdentityIdentitydense_3/Sigmoid:y:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?$
?
K__inference_autoencoder_1_layer_call_and_return_conditional_losses_19224132
input_1)
sequential_2_19224107:
??$
sequential_2_19224109:	?)
sequential_3_19224113:
??$
sequential_3_19224115:	?
identity

identity_1??0dense_2/kernel/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?$sequential_2/StatefulPartitionedCall?$sequential_3/StatefulPartitionedCall?
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_2_19224107sequential_2_19224109*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_192237702&
$sequential_2/StatefulPartitionedCall?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_19224113sequential_3_19224115*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_192239162&
$sequential_3/StatefulPartitionedCall?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_2_19224107* 
_output_shapes
:
??*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_3_19224113* 
_output_shapes
:
??*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:01^dense_2/kernel/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity-sequential_2/StatefulPartitionedCall:output:11^dense_2/kernel/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
$__inference__traced_restore_19224691
file_prefix3
assignvariableop_dense_2_kernel:
??.
assignvariableop_1_dense_2_bias:	?5
!assignvariableop_2_dense_3_kernel:
??.
assignvariableop_3_dense_3_bias:	?

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
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_3_biasIdentity_3:output:0"/device:CPU:0*
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
?
?
E__inference_dense_3_layer_call_and_return_conditional_losses_19224606

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoid?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
0__inference_autoencoder_1_layer_call_fn_19224187
x
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_autoencoder_1_layer_call_and_return_conditional_losses_192240502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_nameX
?
?
J__inference_sequential_3_layer_call_and_return_conditional_losses_19224520
dense_3_input:
&dense_3_matmul_readvariableop_resource:
??6
'dense_3_biasadd_readvariableop_resource:	?
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_3_input%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/BiasAddz
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_3/Sigmoid?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
IdentityIdentitydense_3/Sigmoid:y:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp:W S
(
_output_shapes
:??????????
'
_user_specified_namedense_3_input
?
?
/__inference_sequential_3_layer_call_fn_19224460

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_192239162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_2_layer_call_fn_19224323

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_192237042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_dense_3_layer_call_and_return_conditional_losses_19223860

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoid?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_dense_3_layer_call_fn_19224589

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_192238602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_2_layer_call_fn_19224333

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_192237702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_2_layer_call_fn_19223788
input_2
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_192237702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_2
?
?
I__inference_dense_2_layer_call_and_return_all_conditional_losses_19224563

inputs
unknown:
??
	unknown_0:	?
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_192236822
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
GPU 2J 8? *:
f5R3
1__inference_dense_2_activity_regularizer_192236582
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_dense_2_layer_call_fn_19224552

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_192236822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?A
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_19224427

inputs:
&dense_2_matmul_readvariableop_resource:
??6
'dense_2_biasadd_readvariableop_resource:	?
identity

identity_1??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddz
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_2/Sigmoid?
#dense_2/ActivityRegularizer/SigmoidSigmoiddense_2/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2%
#dense_2/ActivityRegularizer/Sigmoid?
2dense_2/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 24
2dense_2/ActivityRegularizer/Mean/reduction_indices?
 dense_2/ActivityRegularizer/MeanMean'dense_2/ActivityRegularizer/Sigmoid:y:0;dense_2/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes	
:?2"
 dense_2/ActivityRegularizer/Mean?
%dense_2/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2'
%dense_2/ActivityRegularizer/Maximum/y?
#dense_2/ActivityRegularizer/MaximumMaximum)dense_2/ActivityRegularizer/Mean:output:0.dense_2/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes	
:?2%
#dense_2/ActivityRegularizer/Maximum?
%dense_2/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2'
%dense_2/ActivityRegularizer/truediv/x?
#dense_2/ActivityRegularizer/truedivRealDiv.dense_2/ActivityRegularizer/truediv/x:output:0'dense_2/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2%
#dense_2/ActivityRegularizer/truediv?
dense_2/ActivityRegularizer/LogLog'dense_2/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes	
:?2!
dense_2/ActivityRegularizer/Log?
!dense_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_2/ActivityRegularizer/mul/x?
dense_2/ActivityRegularizer/mulMul*dense_2/ActivityRegularizer/mul/x:output:0#dense_2/ActivityRegularizer/Log:y:0*
T0*
_output_shapes	
:?2!
dense_2/ActivityRegularizer/mul?
!dense_2/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!dense_2/ActivityRegularizer/sub/x?
dense_2/ActivityRegularizer/subSub*dense_2/ActivityRegularizer/sub/x:output:0'dense_2/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2!
dense_2/ActivityRegularizer/sub?
'dense_2/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2)
'dense_2/ActivityRegularizer/truediv_1/x?
%dense_2/ActivityRegularizer/truediv_1RealDiv0dense_2/ActivityRegularizer/truediv_1/x:output:0#dense_2/ActivityRegularizer/sub:z:0*
T0*
_output_shapes	
:?2'
%dense_2/ActivityRegularizer/truediv_1?
!dense_2/ActivityRegularizer/Log_1Log)dense_2/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes	
:?2#
!dense_2/ActivityRegularizer/Log_1?
#dense_2/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2%
#dense_2/ActivityRegularizer/mul_1/x?
!dense_2/ActivityRegularizer/mul_1Mul,dense_2/ActivityRegularizer/mul_1/x:output:0%dense_2/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes	
:?2#
!dense_2/ActivityRegularizer/mul_1?
dense_2/ActivityRegularizer/addAddV2#dense_2/ActivityRegularizer/mul:z:0%dense_2/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes	
:?2!
dense_2/ActivityRegularizer/add?
!dense_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_2/ActivityRegularizer/Const?
dense_2/ActivityRegularizer/SumSum#dense_2/ActivityRegularizer/add:z:0*dense_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_2/ActivityRegularizer/Sum?
#dense_2/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#dense_2/ActivityRegularizer/mul_2/x?
!dense_2/ActivityRegularizer/mul_2Mul,dense_2/ActivityRegularizer/mul_2/x:output:0(dense_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_2/ActivityRegularizer/mul_2?
!dense_2/ActivityRegularizer/ShapeShapedense_2/Sigmoid:y:0*
T0*
_output_shapes
:2#
!dense_2/ActivityRegularizer/Shape?
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_2/ActivityRegularizer/strided_slice/stack?
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_1?
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_2?
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_2/ActivityRegularizer/strided_slice?
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_2/ActivityRegularizer/Cast?
%dense_2/ActivityRegularizer/truediv_2RealDiv%dense_2/ActivityRegularizer/mul_2:z:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_2/ActivityRegularizer/truediv_2?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
IdentityIdentitydense_2/Sigmoid:y:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity)dense_2/ActivityRegularizer/truediv_2:z:0^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
0__inference_autoencoder_1_layer_call_fn_19224173
x
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_autoencoder_1_layer_call_and_return_conditional_losses_192239942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_nameX
?d
?
K__inference_autoencoder_1_layer_call_and_return_conditional_losses_19224307
xG
3sequential_2_dense_2_matmul_readvariableop_resource:
??C
4sequential_2_dense_2_biasadd_readvariableop_resource:	?G
3sequential_3_dense_3_matmul_readvariableop_resource:
??C
4sequential_3_dense_3_biasadd_readvariableop_resource:	?
identity

identity_1??0dense_2/kernel/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?+sequential_2/dense_2/BiasAdd/ReadVariableOp?*sequential_2/dense_2/MatMul/ReadVariableOp?+sequential_3/dense_3/BiasAdd/ReadVariableOp?*sequential_3/dense_3/MatMul/ReadVariableOp?
*sequential_2/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential_2/dense_2/MatMul/ReadVariableOp?
sequential_2/dense_2/MatMulMatMulx2sequential_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_2/dense_2/MatMul?
+sequential_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_2/dense_2/BiasAdd/ReadVariableOp?
sequential_2/dense_2/BiasAddBiasAdd%sequential_2/dense_2/MatMul:product:03sequential_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_2/dense_2/BiasAdd?
sequential_2/dense_2/SigmoidSigmoid%sequential_2/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_2/dense_2/Sigmoid?
0sequential_2/dense_2/ActivityRegularizer/SigmoidSigmoid sequential_2/dense_2/Sigmoid:y:0*
T0*(
_output_shapes
:??????????22
0sequential_2/dense_2/ActivityRegularizer/Sigmoid?
?sequential_2/dense_2/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2A
?sequential_2/dense_2/ActivityRegularizer/Mean/reduction_indices?
-sequential_2/dense_2/ActivityRegularizer/MeanMean4sequential_2/dense_2/ActivityRegularizer/Sigmoid:y:0Hsequential_2/dense_2/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes	
:?2/
-sequential_2/dense_2/ActivityRegularizer/Mean?
2sequential_2/dense_2/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.24
2sequential_2/dense_2/ActivityRegularizer/Maximum/y?
0sequential_2/dense_2/ActivityRegularizer/MaximumMaximum6sequential_2/dense_2/ActivityRegularizer/Mean:output:0;sequential_2/dense_2/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes	
:?22
0sequential_2/dense_2/ActivityRegularizer/Maximum?
2sequential_2/dense_2/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<24
2sequential_2/dense_2/ActivityRegularizer/truediv/x?
0sequential_2/dense_2/ActivityRegularizer/truedivRealDiv;sequential_2/dense_2/ActivityRegularizer/truediv/x:output:04sequential_2/dense_2/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?22
0sequential_2/dense_2/ActivityRegularizer/truediv?
,sequential_2/dense_2/ActivityRegularizer/LogLog4sequential_2/dense_2/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes	
:?2.
,sequential_2/dense_2/ActivityRegularizer/Log?
.sequential_2/dense_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.sequential_2/dense_2/ActivityRegularizer/mul/x?
,sequential_2/dense_2/ActivityRegularizer/mulMul7sequential_2/dense_2/ActivityRegularizer/mul/x:output:00sequential_2/dense_2/ActivityRegularizer/Log:y:0*
T0*
_output_shapes	
:?2.
,sequential_2/dense_2/ActivityRegularizer/mul?
.sequential_2/dense_2/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??20
.sequential_2/dense_2/ActivityRegularizer/sub/x?
,sequential_2/dense_2/ActivityRegularizer/subSub7sequential_2/dense_2/ActivityRegularizer/sub/x:output:04sequential_2/dense_2/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2.
,sequential_2/dense_2/ActivityRegularizer/sub?
4sequential_2/dense_2/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?26
4sequential_2/dense_2/ActivityRegularizer/truediv_1/x?
2sequential_2/dense_2/ActivityRegularizer/truediv_1RealDiv=sequential_2/dense_2/ActivityRegularizer/truediv_1/x:output:00sequential_2/dense_2/ActivityRegularizer/sub:z:0*
T0*
_output_shapes	
:?24
2sequential_2/dense_2/ActivityRegularizer/truediv_1?
.sequential_2/dense_2/ActivityRegularizer/Log_1Log6sequential_2/dense_2/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes	
:?20
.sequential_2/dense_2/ActivityRegularizer/Log_1?
0sequential_2/dense_2/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?22
0sequential_2/dense_2/ActivityRegularizer/mul_1/x?
.sequential_2/dense_2/ActivityRegularizer/mul_1Mul9sequential_2/dense_2/ActivityRegularizer/mul_1/x:output:02sequential_2/dense_2/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes	
:?20
.sequential_2/dense_2/ActivityRegularizer/mul_1?
,sequential_2/dense_2/ActivityRegularizer/addAddV20sequential_2/dense_2/ActivityRegularizer/mul:z:02sequential_2/dense_2/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes	
:?2.
,sequential_2/dense_2/ActivityRegularizer/add?
.sequential_2/dense_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_2/dense_2/ActivityRegularizer/Const?
,sequential_2/dense_2/ActivityRegularizer/SumSum0sequential_2/dense_2/ActivityRegularizer/add:z:07sequential_2/dense_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2.
,sequential_2/dense_2/ActivityRegularizer/Sum?
0sequential_2/dense_2/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??22
0sequential_2/dense_2/ActivityRegularizer/mul_2/x?
.sequential_2/dense_2/ActivityRegularizer/mul_2Mul9sequential_2/dense_2/ActivityRegularizer/mul_2/x:output:05sequential_2/dense_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 20
.sequential_2/dense_2/ActivityRegularizer/mul_2?
.sequential_2/dense_2/ActivityRegularizer/ShapeShape sequential_2/dense_2/Sigmoid:y:0*
T0*
_output_shapes
:20
.sequential_2/dense_2/ActivityRegularizer/Shape?
<sequential_2/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential_2/dense_2/ActivityRegularizer/strided_slice/stack?
>sequential_2/dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_2/dense_2/ActivityRegularizer/strided_slice/stack_1?
>sequential_2/dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_2/dense_2/ActivityRegularizer/strided_slice/stack_2?
6sequential_2/dense_2/ActivityRegularizer/strided_sliceStridedSlice7sequential_2/dense_2/ActivityRegularizer/Shape:output:0Esequential_2/dense_2/ActivityRegularizer/strided_slice/stack:output:0Gsequential_2/dense_2/ActivityRegularizer/strided_slice/stack_1:output:0Gsequential_2/dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6sequential_2/dense_2/ActivityRegularizer/strided_slice?
-sequential_2/dense_2/ActivityRegularizer/CastCast?sequential_2/dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-sequential_2/dense_2/ActivityRegularizer/Cast?
2sequential_2/dense_2/ActivityRegularizer/truediv_2RealDiv2sequential_2/dense_2/ActivityRegularizer/mul_2:z:01sequential_2/dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 24
2sequential_2/dense_2/ActivityRegularizer/truediv_2?
*sequential_3/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential_3/dense_3/MatMul/ReadVariableOp?
sequential_3/dense_3/MatMulMatMul sequential_2/dense_2/Sigmoid:y:02sequential_3/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_3/MatMul?
+sequential_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_3/dense_3/BiasAdd/ReadVariableOp?
sequential_3/dense_3/BiasAddBiasAdd%sequential_3/dense_3/MatMul:product:03sequential_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_3/BiasAdd?
sequential_3/dense_3/SigmoidSigmoid%sequential_3/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_3/Sigmoid?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_2_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
IdentityIdentity sequential_3/dense_3/Sigmoid:y:01^dense_2/kernel/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp,^sequential_2/dense_2/BiasAdd/ReadVariableOp+^sequential_2/dense_2/MatMul/ReadVariableOp,^sequential_3/dense_3/BiasAdd/ReadVariableOp+^sequential_3/dense_3/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity6sequential_2/dense_2/ActivityRegularizer/truediv_2:z:01^dense_2/kernel/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp,^sequential_2/dense_2/BiasAdd/ReadVariableOp+^sequential_2/dense_2/MatMul/ReadVariableOp,^sequential_3/dense_3/BiasAdd/ReadVariableOp+^sequential_3/dense_3/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2Z
+sequential_2/dense_2/BiasAdd/ReadVariableOp+sequential_2/dense_2/BiasAdd/ReadVariableOp2X
*sequential_2/dense_2/MatMul/ReadVariableOp*sequential_2/dense_2/MatMul/ReadVariableOp2Z
+sequential_3/dense_3/BiasAdd/ReadVariableOp+sequential_3/dense_3/BiasAdd/ReadVariableOp2X
*sequential_3/dense_3/MatMul/ReadVariableOp*sequential_3/dense_3/MatMul/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_nameX
?
?
__inference_loss_fn_0_19224574M
9dense_2_kernel_regularizer_square_readvariableop_resource:
??
identity??0dense_2/kernel/Regularizer/Square/ReadVariableOp?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_2_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
IdentityIdentity"dense_2/kernel/Regularizer/mul:z:01^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp
?"
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_19223812
input_2$
dense_2_19223791:
??
dense_2_19223793:	?
identity

identity_1??dense_2/StatefulPartitionedCall?0dense_2/kernel/Regularizer/Square/ReadVariableOp?
dense_2/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_2_19223791dense_2_19223793*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_192236822!
dense_2/StatefulPartitionedCall?
+dense_2/ActivityRegularizer/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *:
f5R3
1__inference_dense_2_activity_regularizer_192236582-
+dense_2/ActivityRegularizer/PartitionedCall?
!dense_2/ActivityRegularizer/ShapeShape(dense_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_2/ActivityRegularizer/Shape?
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_2/ActivityRegularizer/strided_slice/stack?
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_1?
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_2?
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_2/ActivityRegularizer/strided_slice?
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_2/ActivityRegularizer/Cast?
#dense_2/ActivityRegularizer/truedivRealDiv4dense_2/ActivityRegularizer/PartitionedCall:output:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_2/ActivityRegularizer/truediv?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_19223791* 
_output_shapes
:
??*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity'dense_2/ActivityRegularizer/truediv:z:0 ^dense_2/StatefulPartitionedCall1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_2
?
?
J__inference_sequential_3_layer_call_and_return_conditional_losses_19223916

inputs$
dense_3_19223904:
??
dense_3_19223906:	?
identity??dense_3/StatefulPartitionedCall?0dense_3/kernel/Regularizer/Square/ReadVariableOp?
dense_3/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_19223904dense_3_19223906*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_3_layer_call_and_return_conditional_losses_192238602!
dense_3/StatefulPartitionedCall?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_3_19223904* 
_output_shapes
:
??*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall1^dense_3/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
Q
1__inference_dense_2_activity_regularizer_19223658

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
?d
?
K__inference_autoencoder_1_layer_call_and_return_conditional_losses_19224247
xG
3sequential_2_dense_2_matmul_readvariableop_resource:
??C
4sequential_2_dense_2_biasadd_readvariableop_resource:	?G
3sequential_3_dense_3_matmul_readvariableop_resource:
??C
4sequential_3_dense_3_biasadd_readvariableop_resource:	?
identity

identity_1??0dense_2/kernel/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?+sequential_2/dense_2/BiasAdd/ReadVariableOp?*sequential_2/dense_2/MatMul/ReadVariableOp?+sequential_3/dense_3/BiasAdd/ReadVariableOp?*sequential_3/dense_3/MatMul/ReadVariableOp?
*sequential_2/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential_2/dense_2/MatMul/ReadVariableOp?
sequential_2/dense_2/MatMulMatMulx2sequential_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_2/dense_2/MatMul?
+sequential_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_2/dense_2/BiasAdd/ReadVariableOp?
sequential_2/dense_2/BiasAddBiasAdd%sequential_2/dense_2/MatMul:product:03sequential_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_2/dense_2/BiasAdd?
sequential_2/dense_2/SigmoidSigmoid%sequential_2/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_2/dense_2/Sigmoid?
0sequential_2/dense_2/ActivityRegularizer/SigmoidSigmoid sequential_2/dense_2/Sigmoid:y:0*
T0*(
_output_shapes
:??????????22
0sequential_2/dense_2/ActivityRegularizer/Sigmoid?
?sequential_2/dense_2/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2A
?sequential_2/dense_2/ActivityRegularizer/Mean/reduction_indices?
-sequential_2/dense_2/ActivityRegularizer/MeanMean4sequential_2/dense_2/ActivityRegularizer/Sigmoid:y:0Hsequential_2/dense_2/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes	
:?2/
-sequential_2/dense_2/ActivityRegularizer/Mean?
2sequential_2/dense_2/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.24
2sequential_2/dense_2/ActivityRegularizer/Maximum/y?
0sequential_2/dense_2/ActivityRegularizer/MaximumMaximum6sequential_2/dense_2/ActivityRegularizer/Mean:output:0;sequential_2/dense_2/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes	
:?22
0sequential_2/dense_2/ActivityRegularizer/Maximum?
2sequential_2/dense_2/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<24
2sequential_2/dense_2/ActivityRegularizer/truediv/x?
0sequential_2/dense_2/ActivityRegularizer/truedivRealDiv;sequential_2/dense_2/ActivityRegularizer/truediv/x:output:04sequential_2/dense_2/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?22
0sequential_2/dense_2/ActivityRegularizer/truediv?
,sequential_2/dense_2/ActivityRegularizer/LogLog4sequential_2/dense_2/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes	
:?2.
,sequential_2/dense_2/ActivityRegularizer/Log?
.sequential_2/dense_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.sequential_2/dense_2/ActivityRegularizer/mul/x?
,sequential_2/dense_2/ActivityRegularizer/mulMul7sequential_2/dense_2/ActivityRegularizer/mul/x:output:00sequential_2/dense_2/ActivityRegularizer/Log:y:0*
T0*
_output_shapes	
:?2.
,sequential_2/dense_2/ActivityRegularizer/mul?
.sequential_2/dense_2/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??20
.sequential_2/dense_2/ActivityRegularizer/sub/x?
,sequential_2/dense_2/ActivityRegularizer/subSub7sequential_2/dense_2/ActivityRegularizer/sub/x:output:04sequential_2/dense_2/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2.
,sequential_2/dense_2/ActivityRegularizer/sub?
4sequential_2/dense_2/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?26
4sequential_2/dense_2/ActivityRegularizer/truediv_1/x?
2sequential_2/dense_2/ActivityRegularizer/truediv_1RealDiv=sequential_2/dense_2/ActivityRegularizer/truediv_1/x:output:00sequential_2/dense_2/ActivityRegularizer/sub:z:0*
T0*
_output_shapes	
:?24
2sequential_2/dense_2/ActivityRegularizer/truediv_1?
.sequential_2/dense_2/ActivityRegularizer/Log_1Log6sequential_2/dense_2/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes	
:?20
.sequential_2/dense_2/ActivityRegularizer/Log_1?
0sequential_2/dense_2/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?22
0sequential_2/dense_2/ActivityRegularizer/mul_1/x?
.sequential_2/dense_2/ActivityRegularizer/mul_1Mul9sequential_2/dense_2/ActivityRegularizer/mul_1/x:output:02sequential_2/dense_2/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes	
:?20
.sequential_2/dense_2/ActivityRegularizer/mul_1?
,sequential_2/dense_2/ActivityRegularizer/addAddV20sequential_2/dense_2/ActivityRegularizer/mul:z:02sequential_2/dense_2/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes	
:?2.
,sequential_2/dense_2/ActivityRegularizer/add?
.sequential_2/dense_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_2/dense_2/ActivityRegularizer/Const?
,sequential_2/dense_2/ActivityRegularizer/SumSum0sequential_2/dense_2/ActivityRegularizer/add:z:07sequential_2/dense_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2.
,sequential_2/dense_2/ActivityRegularizer/Sum?
0sequential_2/dense_2/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??22
0sequential_2/dense_2/ActivityRegularizer/mul_2/x?
.sequential_2/dense_2/ActivityRegularizer/mul_2Mul9sequential_2/dense_2/ActivityRegularizer/mul_2/x:output:05sequential_2/dense_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 20
.sequential_2/dense_2/ActivityRegularizer/mul_2?
.sequential_2/dense_2/ActivityRegularizer/ShapeShape sequential_2/dense_2/Sigmoid:y:0*
T0*
_output_shapes
:20
.sequential_2/dense_2/ActivityRegularizer/Shape?
<sequential_2/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential_2/dense_2/ActivityRegularizer/strided_slice/stack?
>sequential_2/dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_2/dense_2/ActivityRegularizer/strided_slice/stack_1?
>sequential_2/dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_2/dense_2/ActivityRegularizer/strided_slice/stack_2?
6sequential_2/dense_2/ActivityRegularizer/strided_sliceStridedSlice7sequential_2/dense_2/ActivityRegularizer/Shape:output:0Esequential_2/dense_2/ActivityRegularizer/strided_slice/stack:output:0Gsequential_2/dense_2/ActivityRegularizer/strided_slice/stack_1:output:0Gsequential_2/dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6sequential_2/dense_2/ActivityRegularizer/strided_slice?
-sequential_2/dense_2/ActivityRegularizer/CastCast?sequential_2/dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-sequential_2/dense_2/ActivityRegularizer/Cast?
2sequential_2/dense_2/ActivityRegularizer/truediv_2RealDiv2sequential_2/dense_2/ActivityRegularizer/mul_2:z:01sequential_2/dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 24
2sequential_2/dense_2/ActivityRegularizer/truediv_2?
*sequential_3/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential_3/dense_3/MatMul/ReadVariableOp?
sequential_3/dense_3/MatMulMatMul sequential_2/dense_2/Sigmoid:y:02sequential_3/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_3/MatMul?
+sequential_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_3/dense_3/BiasAdd/ReadVariableOp?
sequential_3/dense_3/BiasAddBiasAdd%sequential_3/dense_3/MatMul:product:03sequential_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_3/BiasAdd?
sequential_3/dense_3/SigmoidSigmoid%sequential_3/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_3/dense_3/Sigmoid?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_2_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
IdentityIdentity sequential_3/dense_3/Sigmoid:y:01^dense_2/kernel/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp,^sequential_2/dense_2/BiasAdd/ReadVariableOp+^sequential_2/dense_2/MatMul/ReadVariableOp,^sequential_3/dense_3/BiasAdd/ReadVariableOp+^sequential_3/dense_3/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity6sequential_2/dense_2/ActivityRegularizer/truediv_2:z:01^dense_2/kernel/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp,^sequential_2/dense_2/BiasAdd/ReadVariableOp+^sequential_2/dense_2/MatMul/ReadVariableOp,^sequential_3/dense_3/BiasAdd/ReadVariableOp+^sequential_3/dense_3/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2Z
+sequential_2/dense_2/BiasAdd/ReadVariableOp+sequential_2/dense_2/BiasAdd/ReadVariableOp2X
*sequential_2/dense_2/MatMul/ReadVariableOp*sequential_2/dense_2/MatMul/ReadVariableOp2Z
+sequential_3/dense_3/BiasAdd/ReadVariableOp+sequential_3/dense_3/BiasAdd/ReadVariableOp2X
*sequential_3/dense_3/MatMul/ReadVariableOp*sequential_3/dense_3/MatMul/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_nameX
?"
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_19223770

inputs$
dense_2_19223749:
??
dense_2_19223751:	?
identity

identity_1??dense_2/StatefulPartitionedCall?0dense_2/kernel/Regularizer/Square/ReadVariableOp?
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_19223749dense_2_19223751*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_192236822!
dense_2/StatefulPartitionedCall?
+dense_2/ActivityRegularizer/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *:
f5R3
1__inference_dense_2_activity_regularizer_192236582-
+dense_2/ActivityRegularizer/PartitionedCall?
!dense_2/ActivityRegularizer/ShapeShape(dense_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_2/ActivityRegularizer/Shape?
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_2/ActivityRegularizer/strided_slice/stack?
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_1?
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_2?
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_2/ActivityRegularizer/strided_slice?
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_2/ActivityRegularizer/Cast?
#dense_2/ActivityRegularizer/truedivRealDiv4dense_2/ActivityRegularizer/PartitionedCall:output:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_2/ActivityRegularizer/truediv?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_19223749* 
_output_shapes
:
??*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity'dense_2/ActivityRegularizer/truediv:z:0 ^dense_2/StatefulPartitionedCall1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?"
?
J__inference_sequential_2_layer_call_and_return_conditional_losses_19223836
input_2$
dense_2_19223815:
??
dense_2_19223817:	?
identity

identity_1??dense_2/StatefulPartitionedCall?0dense_2/kernel/Regularizer/Square/ReadVariableOp?
dense_2/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_2_19223815dense_2_19223817*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_192236822!
dense_2/StatefulPartitionedCall?
+dense_2/ActivityRegularizer/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *:
f5R3
1__inference_dense_2_activity_regularizer_192236582-
+dense_2/ActivityRegularizer/PartitionedCall?
!dense_2/ActivityRegularizer/ShapeShape(dense_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_2/ActivityRegularizer/Shape?
/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_2/ActivityRegularizer/strided_slice/stack?
1dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_1?
1dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_2/ActivityRegularizer/strided_slice/stack_2?
)dense_2/ActivityRegularizer/strided_sliceStridedSlice*dense_2/ActivityRegularizer/Shape:output:08dense_2/ActivityRegularizer/strided_slice/stack:output:0:dense_2/ActivityRegularizer/strided_slice/stack_1:output:0:dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_2/ActivityRegularizer/strided_slice?
 dense_2/ActivityRegularizer/CastCast2dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_2/ActivityRegularizer/Cast?
#dense_2/ActivityRegularizer/truedivRealDiv4dense_2/ActivityRegularizer/PartitionedCall:output:0$dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_2/ActivityRegularizer/truediv?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_19223815* 
_output_shapes
:
??*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0 ^dense_2/StatefulPartitionedCall1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity'dense_2/ActivityRegularizer/truediv:z:0 ^dense_2/StatefulPartitionedCall1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_2
?
?
!__inference__traced_save_19224669
file_prefix-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableopsavev2_const"/device:CPU:0*
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

identity_1Identity_1:output:0*=
_input_shapes,
*: :
??:?:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: 
?$
?
K__inference_autoencoder_1_layer_call_and_return_conditional_losses_19223994
x)
sequential_2_19223969:
??$
sequential_2_19223971:	?)
sequential_3_19223975:
??$
sequential_3_19223977:	?
identity

identity_1??0dense_2/kernel/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?$sequential_2/StatefulPartitionedCall?$sequential_3/StatefulPartitionedCall?
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallxsequential_2_19223969sequential_2_19223971*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_192237042&
$sequential_2/StatefulPartitionedCall?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_19223975sequential_3_19223977*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_192238732&
$sequential_3/StatefulPartitionedCall?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_2_19223969* 
_output_shapes
:
??*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_3_19223975* 
_output_shapes
:
??*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:01^dense_2/kernel/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity-sequential_2/StatefulPartitionedCall:output:11^dense_2/kernel/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_nameX
?
?
E__inference_dense_2_layer_call_and_return_conditional_losses_19224634

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoid?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?$
?
K__inference_autoencoder_1_layer_call_and_return_conditional_losses_19224104
input_1)
sequential_2_19224079:
??$
sequential_2_19224081:	?)
sequential_3_19224085:
??$
sequential_3_19224087:	?
identity

identity_1??0dense_2/kernel/Regularizer/Square/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?$sequential_2/StatefulPartitionedCall?$sequential_3/StatefulPartitionedCall?
$sequential_2/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_2_19224079sequential_2_19224081*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_192237042&
$sequential_2/StatefulPartitionedCall?
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_19224085sequential_3_19224087*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_3_layer_call_and_return_conditional_losses_192238732&
$sequential_3/StatefulPartitionedCall?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_2_19224079* 
_output_shapes
:
??*
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_2/kernel/Regularizer/Square?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_2/kernel/Regularizer/Const?
dense_2/kernel/Regularizer/SumSum%dense_2/kernel/Regularizer/Square:y:0)dense_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_3_19224085* 
_output_shapes
:
??*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
IdentityIdentity-sequential_3/StatefulPartitionedCall:output:01^dense_2/kernel/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity-sequential_2/StatefulPartitionedCall:output:11^dense_2/kernel/Regularizer/Square/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
J__inference_sequential_3_layer_call_and_return_conditional_losses_19224537
dense_3_input:
&dense_3_matmul_readvariableop_resource:
??6
'dense_3_biasadd_readvariableop_resource:	?
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_3_input%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/BiasAddz
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_3/Sigmoid?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
IdentityIdentitydense_3/Sigmoid:y:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp:W S
(
_output_shapes
:??????????
'
_user_specified_namedense_3_input
?
?
/__inference_sequential_2_layer_call_fn_19223712
input_2
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_2_layer_call_and_return_conditional_losses_192237042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_2
?\
?
#__inference__wrapped_model_19223628
input_1U
Aautoencoder_1_sequential_2_dense_2_matmul_readvariableop_resource:
??Q
Bautoencoder_1_sequential_2_dense_2_biasadd_readvariableop_resource:	?U
Aautoencoder_1_sequential_3_dense_3_matmul_readvariableop_resource:
??Q
Bautoencoder_1_sequential_3_dense_3_biasadd_readvariableop_resource:	?
identity??9autoencoder_1/sequential_2/dense_2/BiasAdd/ReadVariableOp?8autoencoder_1/sequential_2/dense_2/MatMul/ReadVariableOp?9autoencoder_1/sequential_3/dense_3/BiasAdd/ReadVariableOp?8autoencoder_1/sequential_3/dense_3/MatMul/ReadVariableOp?
8autoencoder_1/sequential_2/dense_2/MatMul/ReadVariableOpReadVariableOpAautoencoder_1_sequential_2_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02:
8autoencoder_1/sequential_2/dense_2/MatMul/ReadVariableOp?
)autoencoder_1/sequential_2/dense_2/MatMulMatMulinput_1@autoencoder_1/sequential_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)autoencoder_1/sequential_2/dense_2/MatMul?
9autoencoder_1/sequential_2/dense_2/BiasAdd/ReadVariableOpReadVariableOpBautoencoder_1_sequential_2_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02;
9autoencoder_1/sequential_2/dense_2/BiasAdd/ReadVariableOp?
*autoencoder_1/sequential_2/dense_2/BiasAddBiasAdd3autoencoder_1/sequential_2/dense_2/MatMul:product:0Aautoencoder_1/sequential_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*autoencoder_1/sequential_2/dense_2/BiasAdd?
*autoencoder_1/sequential_2/dense_2/SigmoidSigmoid3autoencoder_1/sequential_2/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2,
*autoencoder_1/sequential_2/dense_2/Sigmoid?
>autoencoder_1/sequential_2/dense_2/ActivityRegularizer/SigmoidSigmoid.autoencoder_1/sequential_2/dense_2/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2@
>autoencoder_1/sequential_2/dense_2/ActivityRegularizer/Sigmoid?
Mautoencoder_1/sequential_2/dense_2/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mautoencoder_1/sequential_2/dense_2/ActivityRegularizer/Mean/reduction_indices?
;autoencoder_1/sequential_2/dense_2/ActivityRegularizer/MeanMeanBautoencoder_1/sequential_2/dense_2/ActivityRegularizer/Sigmoid:y:0Vautoencoder_1/sequential_2/dense_2/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes	
:?2=
;autoencoder_1/sequential_2/dense_2/ActivityRegularizer/Mean?
@autoencoder_1/sequential_2/dense_2/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2B
@autoencoder_1/sequential_2/dense_2/ActivityRegularizer/Maximum/y?
>autoencoder_1/sequential_2/dense_2/ActivityRegularizer/MaximumMaximumDautoencoder_1/sequential_2/dense_2/ActivityRegularizer/Mean:output:0Iautoencoder_1/sequential_2/dense_2/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes	
:?2@
>autoencoder_1/sequential_2/dense_2/ActivityRegularizer/Maximum?
@autoencoder_1/sequential_2/dense_2/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2B
@autoencoder_1/sequential_2/dense_2/ActivityRegularizer/truediv/x?
>autoencoder_1/sequential_2/dense_2/ActivityRegularizer/truedivRealDivIautoencoder_1/sequential_2/dense_2/ActivityRegularizer/truediv/x:output:0Bautoencoder_1/sequential_2/dense_2/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2@
>autoencoder_1/sequential_2/dense_2/ActivityRegularizer/truediv?
:autoencoder_1/sequential_2/dense_2/ActivityRegularizer/LogLogBautoencoder_1/sequential_2/dense_2/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes	
:?2<
:autoencoder_1/sequential_2/dense_2/ActivityRegularizer/Log?
<autoencoder_1/sequential_2/dense_2/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2>
<autoencoder_1/sequential_2/dense_2/ActivityRegularizer/mul/x?
:autoencoder_1/sequential_2/dense_2/ActivityRegularizer/mulMulEautoencoder_1/sequential_2/dense_2/ActivityRegularizer/mul/x:output:0>autoencoder_1/sequential_2/dense_2/ActivityRegularizer/Log:y:0*
T0*
_output_shapes	
:?2<
:autoencoder_1/sequential_2/dense_2/ActivityRegularizer/mul?
<autoencoder_1/sequential_2/dense_2/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2>
<autoencoder_1/sequential_2/dense_2/ActivityRegularizer/sub/x?
:autoencoder_1/sequential_2/dense_2/ActivityRegularizer/subSubEautoencoder_1/sequential_2/dense_2/ActivityRegularizer/sub/x:output:0Bautoencoder_1/sequential_2/dense_2/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2<
:autoencoder_1/sequential_2/dense_2/ActivityRegularizer/sub?
Bautoencoder_1/sequential_2/dense_2/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2D
Bautoencoder_1/sequential_2/dense_2/ActivityRegularizer/truediv_1/x?
@autoencoder_1/sequential_2/dense_2/ActivityRegularizer/truediv_1RealDivKautoencoder_1/sequential_2/dense_2/ActivityRegularizer/truediv_1/x:output:0>autoencoder_1/sequential_2/dense_2/ActivityRegularizer/sub:z:0*
T0*
_output_shapes	
:?2B
@autoencoder_1/sequential_2/dense_2/ActivityRegularizer/truediv_1?
<autoencoder_1/sequential_2/dense_2/ActivityRegularizer/Log_1LogDautoencoder_1/sequential_2/dense_2/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes	
:?2>
<autoencoder_1/sequential_2/dense_2/ActivityRegularizer/Log_1?
>autoencoder_1/sequential_2/dense_2/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2@
>autoencoder_1/sequential_2/dense_2/ActivityRegularizer/mul_1/x?
<autoencoder_1/sequential_2/dense_2/ActivityRegularizer/mul_1MulGautoencoder_1/sequential_2/dense_2/ActivityRegularizer/mul_1/x:output:0@autoencoder_1/sequential_2/dense_2/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes	
:?2>
<autoencoder_1/sequential_2/dense_2/ActivityRegularizer/mul_1?
:autoencoder_1/sequential_2/dense_2/ActivityRegularizer/addAddV2>autoencoder_1/sequential_2/dense_2/ActivityRegularizer/mul:z:0@autoencoder_1/sequential_2/dense_2/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes	
:?2<
:autoencoder_1/sequential_2/dense_2/ActivityRegularizer/add?
<autoencoder_1/sequential_2/dense_2/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2>
<autoencoder_1/sequential_2/dense_2/ActivityRegularizer/Const?
:autoencoder_1/sequential_2/dense_2/ActivityRegularizer/SumSum>autoencoder_1/sequential_2/dense_2/ActivityRegularizer/add:z:0Eautoencoder_1/sequential_2/dense_2/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2<
:autoencoder_1/sequential_2/dense_2/ActivityRegularizer/Sum?
>autoencoder_1/sequential_2/dense_2/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2@
>autoencoder_1/sequential_2/dense_2/ActivityRegularizer/mul_2/x?
<autoencoder_1/sequential_2/dense_2/ActivityRegularizer/mul_2MulGautoencoder_1/sequential_2/dense_2/ActivityRegularizer/mul_2/x:output:0Cautoencoder_1/sequential_2/dense_2/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2>
<autoencoder_1/sequential_2/dense_2/ActivityRegularizer/mul_2?
<autoencoder_1/sequential_2/dense_2/ActivityRegularizer/ShapeShape.autoencoder_1/sequential_2/dense_2/Sigmoid:y:0*
T0*
_output_shapes
:2>
<autoencoder_1/sequential_2/dense_2/ActivityRegularizer/Shape?
Jautoencoder_1/sequential_2/dense_2/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2L
Jautoencoder_1/sequential_2/dense_2/ActivityRegularizer/strided_slice/stack?
Lautoencoder_1/sequential_2/dense_2/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2N
Lautoencoder_1/sequential_2/dense_2/ActivityRegularizer/strided_slice/stack_1?
Lautoencoder_1/sequential_2/dense_2/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2N
Lautoencoder_1/sequential_2/dense_2/ActivityRegularizer/strided_slice/stack_2?
Dautoencoder_1/sequential_2/dense_2/ActivityRegularizer/strided_sliceStridedSliceEautoencoder_1/sequential_2/dense_2/ActivityRegularizer/Shape:output:0Sautoencoder_1/sequential_2/dense_2/ActivityRegularizer/strided_slice/stack:output:0Uautoencoder_1/sequential_2/dense_2/ActivityRegularizer/strided_slice/stack_1:output:0Uautoencoder_1/sequential_2/dense_2/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2F
Dautoencoder_1/sequential_2/dense_2/ActivityRegularizer/strided_slice?
;autoencoder_1/sequential_2/dense_2/ActivityRegularizer/CastCastMautoencoder_1/sequential_2/dense_2/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2=
;autoencoder_1/sequential_2/dense_2/ActivityRegularizer/Cast?
@autoencoder_1/sequential_2/dense_2/ActivityRegularizer/truediv_2RealDiv@autoencoder_1/sequential_2/dense_2/ActivityRegularizer/mul_2:z:0?autoencoder_1/sequential_2/dense_2/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2B
@autoencoder_1/sequential_2/dense_2/ActivityRegularizer/truediv_2?
8autoencoder_1/sequential_3/dense_3/MatMul/ReadVariableOpReadVariableOpAautoencoder_1_sequential_3_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02:
8autoencoder_1/sequential_3/dense_3/MatMul/ReadVariableOp?
)autoencoder_1/sequential_3/dense_3/MatMulMatMul.autoencoder_1/sequential_2/dense_2/Sigmoid:y:0@autoencoder_1/sequential_3/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)autoencoder_1/sequential_3/dense_3/MatMul?
9autoencoder_1/sequential_3/dense_3/BiasAdd/ReadVariableOpReadVariableOpBautoencoder_1_sequential_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02;
9autoencoder_1/sequential_3/dense_3/BiasAdd/ReadVariableOp?
*autoencoder_1/sequential_3/dense_3/BiasAddBiasAdd3autoencoder_1/sequential_3/dense_3/MatMul:product:0Aautoencoder_1/sequential_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*autoencoder_1/sequential_3/dense_3/BiasAdd?
*autoencoder_1/sequential_3/dense_3/SigmoidSigmoid3autoencoder_1/sequential_3/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2,
*autoencoder_1/sequential_3/dense_3/Sigmoid?
IdentityIdentity.autoencoder_1/sequential_3/dense_3/Sigmoid:y:0:^autoencoder_1/sequential_2/dense_2/BiasAdd/ReadVariableOp9^autoencoder_1/sequential_2/dense_2/MatMul/ReadVariableOp:^autoencoder_1/sequential_3/dense_3/BiasAdd/ReadVariableOp9^autoencoder_1/sequential_3/dense_3/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2v
9autoencoder_1/sequential_2/dense_2/BiasAdd/ReadVariableOp9autoencoder_1/sequential_2/dense_2/BiasAdd/ReadVariableOp2t
8autoencoder_1/sequential_2/dense_2/MatMul/ReadVariableOp8autoencoder_1/sequential_2/dense_2/MatMul/ReadVariableOp2v
9autoencoder_1/sequential_3/dense_3/BiasAdd/ReadVariableOp9autoencoder_1/sequential_3/dense_3/BiasAdd/ReadVariableOp2t
8autoencoder_1/sequential_3/dense_3/MatMul/ReadVariableOp8autoencoder_1/sequential_3/dense_3/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
0__inference_autoencoder_1_layer_call_fn_19224006
input_1
unknown:
??
	unknown_0:	?
	unknown_1:
??
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_autoencoder_1_layer_call_and_return_conditional_losses_192239942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
J__inference_sequential_3_layer_call_and_return_conditional_losses_19224486

inputs:
&dense_3_matmul_readvariableop_resource:
??6
'dense_3_biasadd_readvariableop_resource:	?
identity??dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?0dense_3/kernel/Regularizer/Square/ReadVariableOp?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/BiasAddz
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_3/Sigmoid?
0dense_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_3/kernel/Regularizer/Square/ReadVariableOp?
!dense_3/kernel/Regularizer/SquareSquare8dense_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_3/kernel/Regularizer/Square?
 dense_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_3/kernel/Regularizer/Const?
dense_3/kernel/Regularizer/SumSum%dense_3/kernel/Regularizer/Square:y:0)dense_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/Sum?
 dense_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_3/kernel/Regularizer/mul/x?
dense_3/kernel/Regularizer/mulMul)dense_3/kernel/Regularizer/mul/x:output:0'dense_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_3/kernel/Regularizer/mul?
IdentityIdentitydense_3/Sigmoid:y:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp1^dense_3/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2d
0dense_3/kernel/Regularizer/Square/ReadVariableOp0dense_3/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
<
input_11
serving_default_input_1:0??????????=
output_11
StatefulPartitionedCall:0??????????tensorflow/serving/predict:??
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
8__call__
*9&call_and_return_all_conditional_losses
:_default_save_signature"?
_tf_keras_model?{"name": "autoencoder_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Autoencoder", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 512]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Autoencoder"}}
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
;__call__
*<&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 256, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}}, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 6}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 512]}, "float32", "input_2"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 256, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}]}}}
?
layer_with_weights-0
layer-0
	variables
trainable_variables
regularization_losses
	keras_api
=__call__
*>&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_3_input"}}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 512, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 256]}, "float32", "dense_3_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_3_input"}, "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 512, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8}]}}}
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
	variables
non_trainable_variables
layer_metrics
trainable_variables
regularization_losses
metrics

layers
8__call__
:_default_save_signature
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
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
@__call__
*A&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 256, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 6}, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
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

	variables
!non_trainable_variables
"layer_metrics
trainable_variables
regularization_losses
#metrics

$layers
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
?	

kernel
bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
C__call__
*D&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 512, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 256]}}
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
	variables
*non_trainable_variables
+layer_metrics
trainable_variables
regularization_losses
,metrics

-layers
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_2/kernel
:?2dense_2/bias
": 
??2dense_3/kernel
:?2dense_3/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
?
.layer_regularization_losses
	variables
/non_trainable_variables
0layer_metrics
trainable_variables
regularization_losses
1metrics

2layers
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
trackable_dict_wrapper
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
?
3layer_regularization_losses
%	variables
4non_trainable_variables
5layer_metrics
&trainable_variables
'regularization_losses
6metrics

7layers
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
B0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
E0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
0__inference_autoencoder_1_layer_call_fn_19224006
0__inference_autoencoder_1_layer_call_fn_19224173
0__inference_autoencoder_1_layer_call_fn_19224187
0__inference_autoencoder_1_layer_call_fn_19224076?
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
K__inference_autoencoder_1_layer_call_and_return_conditional_losses_19224247
K__inference_autoencoder_1_layer_call_and_return_conditional_losses_19224307
K__inference_autoencoder_1_layer_call_and_return_conditional_losses_19224104
K__inference_autoencoder_1_layer_call_and_return_conditional_losses_19224132?
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
#__inference__wrapped_model_19223628?
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
annotations? *'?$
"?
input_1??????????
?2?
/__inference_sequential_2_layer_call_fn_19223712
/__inference_sequential_2_layer_call_fn_19224323
/__inference_sequential_2_layer_call_fn_19224333
/__inference_sequential_2_layer_call_fn_19223788?
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
J__inference_sequential_2_layer_call_and_return_conditional_losses_19224380
J__inference_sequential_2_layer_call_and_return_conditional_losses_19224427
J__inference_sequential_2_layer_call_and_return_conditional_losses_19223812
J__inference_sequential_2_layer_call_and_return_conditional_losses_19223836?
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
/__inference_sequential_3_layer_call_fn_19224442
/__inference_sequential_3_layer_call_fn_19224451
/__inference_sequential_3_layer_call_fn_19224460
/__inference_sequential_3_layer_call_fn_19224469?
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
J__inference_sequential_3_layer_call_and_return_conditional_losses_19224486
J__inference_sequential_3_layer_call_and_return_conditional_losses_19224503
J__inference_sequential_3_layer_call_and_return_conditional_losses_19224520
J__inference_sequential_3_layer_call_and_return_conditional_losses_19224537?
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
&__inference_signature_wrapper_19224159input_1"?
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
*__inference_dense_2_layer_call_fn_19224552?
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
I__inference_dense_2_layer_call_and_return_all_conditional_losses_19224563?
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
__inference_loss_fn_0_19224574?
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
*__inference_dense_3_layer_call_fn_19224589?
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
E__inference_dense_3_layer_call_and_return_conditional_losses_19224606?
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
__inference_loss_fn_1_19224617?
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
1__inference_dense_2_activity_regularizer_19223658?
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
E__inference_dense_2_layer_call_and_return_conditional_losses_19224634?
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
#__inference__wrapped_model_19223628o1?.
'?$
"?
input_1??????????
? "4?1
/
output_1#? 
output_1???????????
K__inference_autoencoder_1_layer_call_and_return_conditional_losses_19224104s5?2
+?(
"?
input_1??????????
p 
? "4?1
?
0??????????
?
?	
1/0 ?
K__inference_autoencoder_1_layer_call_and_return_conditional_losses_19224132s5?2
+?(
"?
input_1??????????
p
? "4?1
?
0??????????
?
?	
1/0 ?
K__inference_autoencoder_1_layer_call_and_return_conditional_losses_19224247m/?,
%?"
?
X??????????
p 
? "4?1
?
0??????????
?
?	
1/0 ?
K__inference_autoencoder_1_layer_call_and_return_conditional_losses_19224307m/?,
%?"
?
X??????????
p
? "4?1
?
0??????????
?
?	
1/0 ?
0__inference_autoencoder_1_layer_call_fn_19224006X5?2
+?(
"?
input_1??????????
p 
? "????????????
0__inference_autoencoder_1_layer_call_fn_19224076X5?2
+?(
"?
input_1??????????
p
? "????????????
0__inference_autoencoder_1_layer_call_fn_19224173R/?,
%?"
?
X??????????
p 
? "????????????
0__inference_autoencoder_1_layer_call_fn_19224187R/?,
%?"
?
X??????????
p
? "???????????d
1__inference_dense_2_activity_regularizer_19223658/$?!
?
?

activation
? "? ?
I__inference_dense_2_layer_call_and_return_all_conditional_losses_19224563l0?-
&?#
!?
inputs??????????
? "4?1
?
0??????????
?
?	
1/0 ?
E__inference_dense_2_layer_call_and_return_conditional_losses_19224634^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_2_layer_call_fn_19224552Q0?-
&?#
!?
inputs??????????
? "????????????
E__inference_dense_3_layer_call_and_return_conditional_losses_19224606^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_3_layer_call_fn_19224589Q0?-
&?#
!?
inputs??????????
? "???????????=
__inference_loss_fn_0_19224574?

? 
? "? =
__inference_loss_fn_1_19224617?

? 
? "? ?
J__inference_sequential_2_layer_call_and_return_conditional_losses_19223812u9?6
/?,
"?
input_2??????????
p 

 
? "4?1
?
0??????????
?
?	
1/0 ?
J__inference_sequential_2_layer_call_and_return_conditional_losses_19223836u9?6
/?,
"?
input_2??????????
p

 
? "4?1
?
0??????????
?
?	
1/0 ?
J__inference_sequential_2_layer_call_and_return_conditional_losses_19224380t8?5
.?+
!?
inputs??????????
p 

 
? "4?1
?
0??????????
?
?	
1/0 ?
J__inference_sequential_2_layer_call_and_return_conditional_losses_19224427t8?5
.?+
!?
inputs??????????
p

 
? "4?1
?
0??????????
?
?	
1/0 ?
/__inference_sequential_2_layer_call_fn_19223712Z9?6
/?,
"?
input_2??????????
p 

 
? "????????????
/__inference_sequential_2_layer_call_fn_19223788Z9?6
/?,
"?
input_2??????????
p

 
? "????????????
/__inference_sequential_2_layer_call_fn_19224323Y8?5
.?+
!?
inputs??????????
p 

 
? "????????????
/__inference_sequential_2_layer_call_fn_19224333Y8?5
.?+
!?
inputs??????????
p

 
? "????????????
J__inference_sequential_3_layer_call_and_return_conditional_losses_19224486f8?5
.?+
!?
inputs??????????
p 

 
? "&?#
?
0??????????
? ?
J__inference_sequential_3_layer_call_and_return_conditional_losses_19224503f8?5
.?+
!?
inputs??????????
p

 
? "&?#
?
0??????????
? ?
J__inference_sequential_3_layer_call_and_return_conditional_losses_19224520m??<
5?2
(?%
dense_3_input??????????
p 

 
? "&?#
?
0??????????
? ?
J__inference_sequential_3_layer_call_and_return_conditional_losses_19224537m??<
5?2
(?%
dense_3_input??????????
p

 
? "&?#
?
0??????????
? ?
/__inference_sequential_3_layer_call_fn_19224442`??<
5?2
(?%
dense_3_input??????????
p 

 
? "????????????
/__inference_sequential_3_layer_call_fn_19224451Y8?5
.?+
!?
inputs??????????
p 

 
? "????????????
/__inference_sequential_3_layer_call_fn_19224460Y8?5
.?+
!?
inputs??????????
p

 
? "????????????
/__inference_sequential_3_layer_call_fn_19224469`??<
5?2
(?%
dense_3_input??????????
p

 
? "????????????
&__inference_signature_wrapper_19224159z<?9
? 
2?/
-
input_1"?
input_1??????????"4?1
/
output_1#? 
output_1??????????