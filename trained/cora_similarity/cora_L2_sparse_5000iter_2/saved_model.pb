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
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
z
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_6/kernel
s
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel* 
_output_shapes
:
??*
dtype0
q
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes	
:?*
dtype0
z
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_7/kernel
s
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel* 
_output_shapes
:
??*
dtype0
q
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_7/bias
j
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
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
	variables
layer_regularization_losses
metrics

layers
layer_metrics
trainable_variables
regularization_losses
non_trainable_variables
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
 layer_regularization_losses
!metrics

"layers
#layer_metrics
trainable_variables
regularization_losses
$non_trainable_variables
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
)layer_regularization_losses
*metrics

+layers
,layer_metrics
trainable_variables
regularization_losses
-non_trainable_variables
JH
VARIABLE_VALUEdense_6/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_6/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_7/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_7/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
 
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
trainable_variables
.layer_regularization_losses
/metrics
0layer_metrics

1layers
regularization_losses
2non_trainable_variables
 
 

	0
 
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
&trainable_variables
3layer_regularization_losses
4metrics
5layer_metrics

6layers
'regularization_losses
7non_trainable_variables
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
 
|
serving_default_input_1Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_6/kerneldense_6/biasdense_7/kerneldense_7/bias*
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
&__inference_signature_wrapper_14423328
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOpConst*
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
!__inference__traced_save_14423887
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_6/kerneldense_6/biasdense_7/kerneldense_7/bias*
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
$__inference__traced_restore_14423909??
?
?
J__inference_sequential_7_layer_call_and_return_conditional_losses_14423090

inputs$
dense_7_14423084:
??
dense_7_14423086:	?
identity??dense_7/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCallinputsdense_7_14423084dense_7_14423086*
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
E__inference_dense_7_layer_call_and_return_conditional_losses_144230832!
dense_7/StatefulPartitionedCall?
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_7/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_dense_7_layer_call_and_return_conditional_losses_14423083

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
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
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_7_layer_call_and_return_conditional_losses_14423731
dense_7_input:
&dense_7_matmul_readvariableop_resource:
??6
'dense_7_biasadd_readvariableop_resource:	?
identity??dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMuldense_7_input%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_7/BiasAddz
dense_7/SigmoidSigmoiddense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_7/Sigmoid?
IdentityIdentitydense_7/Sigmoid:y:0^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:W S
(
_output_shapes
:??????????
'
_user_specified_namedense_7_input
?
?
0__inference_autoencoder_3_layer_call_fn_14423464
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
K__inference_autoencoder_3_layer_call_and_return_conditional_losses_144232372
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
?
?
/__inference_sequential_7_layer_call_fn_14423769

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
J__inference_sequential_7_layer_call_and_return_conditional_losses_144231272
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
?
Q
1__inference_dense_6_activity_regularizer_14422887

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
?
?
/__inference_sequential_6_layer_call_fn_14423698
dense_6_input
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_6_inputunknown	unknown_0*
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_144229992
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
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_namedense_6_input
?
?
!__inference__traced_save_14423887
file_prefix-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
?
?
$__inference__traced_restore_14423909
file_prefix3
assignvariableop_dense_6_kernel:
??.
assignvariableop_1_dense_6_bias:	?5
!assignvariableop_2_dense_7_kernel:
??.
assignvariableop_3_dense_7_bias:	?

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
AssignVariableOpAssignVariableOpassignvariableop_dense_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_6_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_7_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_7_biasIdentity_3:output:0"/device:CPU:0*
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
?
?
K__inference_autoencoder_3_layer_call_and_return_conditional_losses_14423187
x)
sequential_6_14423168:
??$
sequential_6_14423170:	?)
sequential_7_14423174:
??$
sequential_7_14423176:	?
identity

identity_1??0dense_6/kernel/Regularizer/Square/ReadVariableOp?$sequential_6/StatefulPartitionedCall?$sequential_7/StatefulPartitionedCall?
$sequential_6/StatefulPartitionedCallStatefulPartitionedCallxsequential_6_14423168sequential_6_14423170*
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_144229332&
$sequential_6/StatefulPartitionedCall?
$sequential_7/StatefulPartitionedCallStatefulPartitionedCall-sequential_6/StatefulPartitionedCall:output:0sequential_7_14423174sequential_7_14423176*
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
J__inference_sequential_7_layer_call_and_return_conditional_losses_144230902&
$sequential_7/StatefulPartitionedCall?
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_6_14423168* 
_output_shapes
:
??*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOp?
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_6/kernel/Regularizer/Square?
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/Const?
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/Sum?
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_6/kernel/Regularizer/mul/x?
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mul?
IdentityIdentity-sequential_7/StatefulPartitionedCall:output:01^dense_6/kernel/Regularizer/Square/ReadVariableOp%^sequential_6/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity-sequential_6/StatefulPartitionedCall:output:11^dense_6/kernel/Regularizer/Square/ReadVariableOp%^sequential_6/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall2L
$sequential_7/StatefulPartitionedCall$sequential_7/StatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_nameX
?
?
J__inference_sequential_7_layer_call_and_return_conditional_losses_14423720

inputs:
&dense_7_matmul_readvariableop_resource:
??6
'dense_7_biasadd_readvariableop_resource:	?
identity??dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMulinputs%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_7/BiasAddz
dense_7/SigmoidSigmoiddense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_7/Sigmoid?
IdentityIdentitydense_7/Sigmoid:y:0^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?"
?
J__inference_sequential_6_layer_call_and_return_conditional_losses_14422999

inputs$
dense_6_14422978:
??
dense_6_14422980:	?
identity

identity_1??dense_6/StatefulPartitionedCall?0dense_6/kernel/Regularizer/Square/ReadVariableOp?
dense_6/StatefulPartitionedCallStatefulPartitionedCallinputsdense_6_14422978dense_6_14422980*
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
E__inference_dense_6_layer_call_and_return_conditional_losses_144229112!
dense_6/StatefulPartitionedCall?
+dense_6/ActivityRegularizer/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
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
1__inference_dense_6_activity_regularizer_144228872-
+dense_6/ActivityRegularizer/PartitionedCall?
!dense_6/ActivityRegularizer/ShapeShape(dense_6/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_6/ActivityRegularizer/Shape?
/dense_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_6/ActivityRegularizer/strided_slice/stack?
1dense_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_1?
1dense_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_2?
)dense_6/ActivityRegularizer/strided_sliceStridedSlice*dense_6/ActivityRegularizer/Shape:output:08dense_6/ActivityRegularizer/strided_slice/stack:output:0:dense_6/ActivityRegularizer/strided_slice/stack_1:output:0:dense_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_6/ActivityRegularizer/strided_slice?
 dense_6/ActivityRegularizer/CastCast2dense_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_6/ActivityRegularizer/Cast?
#dense_6/ActivityRegularizer/truedivRealDiv4dense_6/ActivityRegularizer/PartitionedCall:output:0$dense_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_6/ActivityRegularizer/truediv?
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_6_14422978* 
_output_shapes
:
??*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOp?
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_6/kernel/Regularizer/Square?
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/Const?
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/Sum?
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_6/kernel/Regularizer/mul/x?
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mul?
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0 ^dense_6/StatefulPartitionedCall1^dense_6/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity'dense_6/ActivityRegularizer/truediv:z:0 ^dense_6/StatefulPartitionedCall1^dense_6/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
K__inference_autoencoder_3_layer_call_and_return_conditional_losses_14423307
input_1)
sequential_6_14423288:
??$
sequential_6_14423290:	?)
sequential_7_14423294:
??$
sequential_7_14423296:	?
identity

identity_1??0dense_6/kernel/Regularizer/Square/ReadVariableOp?$sequential_6/StatefulPartitionedCall?$sequential_7/StatefulPartitionedCall?
$sequential_6/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_6_14423288sequential_6_14423290*
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_144229992&
$sequential_6/StatefulPartitionedCall?
$sequential_7/StatefulPartitionedCallStatefulPartitionedCall-sequential_6/StatefulPartitionedCall:output:0sequential_7_14423294sequential_7_14423296*
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
J__inference_sequential_7_layer_call_and_return_conditional_losses_144231272&
$sequential_7/StatefulPartitionedCall?
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_6_14423288* 
_output_shapes
:
??*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOp?
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_6/kernel/Regularizer/Square?
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/Const?
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/Sum?
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_6/kernel/Regularizer/mul/x?
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mul?
IdentityIdentity-sequential_7/StatefulPartitionedCall:output:01^dense_6/kernel/Regularizer/Square/ReadVariableOp%^sequential_6/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity-sequential_6/StatefulPartitionedCall:output:11^dense_6/kernel/Regularizer/Square/ReadVariableOp%^sequential_6/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall2L
$sequential_7/StatefulPartitionedCall$sequential_7/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
J__inference_sequential_7_layer_call_and_return_conditional_losses_14423742
dense_7_input:
&dense_7_matmul_readvariableop_resource:
??6
'dense_7_biasadd_readvariableop_resource:	?
identity??dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMuldense_7_input%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_7/BiasAddz
dense_7/SigmoidSigmoiddense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_7/Sigmoid?
IdentityIdentitydense_7/Sigmoid:y:0^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:W S
(
_output_shapes
:??????????
'
_user_specified_namedense_7_input
?"
?
J__inference_sequential_6_layer_call_and_return_conditional_losses_14422933

inputs$
dense_6_14422912:
??
dense_6_14422914:	?
identity

identity_1??dense_6/StatefulPartitionedCall?0dense_6/kernel/Regularizer/Square/ReadVariableOp?
dense_6/StatefulPartitionedCallStatefulPartitionedCallinputsdense_6_14422912dense_6_14422914*
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
E__inference_dense_6_layer_call_and_return_conditional_losses_144229112!
dense_6/StatefulPartitionedCall?
+dense_6/ActivityRegularizer/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
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
1__inference_dense_6_activity_regularizer_144228872-
+dense_6/ActivityRegularizer/PartitionedCall?
!dense_6/ActivityRegularizer/ShapeShape(dense_6/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_6/ActivityRegularizer/Shape?
/dense_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_6/ActivityRegularizer/strided_slice/stack?
1dense_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_1?
1dense_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_2?
)dense_6/ActivityRegularizer/strided_sliceStridedSlice*dense_6/ActivityRegularizer/Shape:output:08dense_6/ActivityRegularizer/strided_slice/stack:output:0:dense_6/ActivityRegularizer/strided_slice/stack_1:output:0:dense_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_6/ActivityRegularizer/strided_slice?
 dense_6/ActivityRegularizer/CastCast2dense_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_6/ActivityRegularizer/Cast?
#dense_6/ActivityRegularizer/truedivRealDiv4dense_6/ActivityRegularizer/PartitionedCall:output:0$dense_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_6/ActivityRegularizer/truediv?
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_6_14422912* 
_output_shapes
:
??*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOp?
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_6/kernel/Regularizer/Square?
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/Const?
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/Sum?
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_6/kernel/Regularizer/mul/x?
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mul?
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0 ^dense_6/StatefulPartitionedCall1^dense_6/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity'dense_6/ActivityRegularizer/truediv:z:0 ^dense_6/StatefulPartitionedCall1^dense_6/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_dense_7_layer_call_fn_14423835

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
E__inference_dense_7_layer_call_and_return_conditional_losses_144230832
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_14423517

inputs:
&dense_6_matmul_readvariableop_resource:
??6
'dense_6_biasadd_readvariableop_resource:	?
identity

identity_1??dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?0dense_6/kernel/Regularizer/Square/ReadVariableOp?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMulinputs%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/BiasAddz
dense_6/SigmoidSigmoiddense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_6/Sigmoid?
#dense_6/ActivityRegularizer/SigmoidSigmoiddense_6/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2%
#dense_6/ActivityRegularizer/Sigmoid?
2dense_6/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 24
2dense_6/ActivityRegularizer/Mean/reduction_indices?
 dense_6/ActivityRegularizer/MeanMean'dense_6/ActivityRegularizer/Sigmoid:y:0;dense_6/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes	
:?2"
 dense_6/ActivityRegularizer/Mean?
%dense_6/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2'
%dense_6/ActivityRegularizer/Maximum/y?
#dense_6/ActivityRegularizer/MaximumMaximum)dense_6/ActivityRegularizer/Mean:output:0.dense_6/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes	
:?2%
#dense_6/ActivityRegularizer/Maximum?
%dense_6/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2'
%dense_6/ActivityRegularizer/truediv/x?
#dense_6/ActivityRegularizer/truedivRealDiv.dense_6/ActivityRegularizer/truediv/x:output:0'dense_6/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2%
#dense_6/ActivityRegularizer/truediv?
dense_6/ActivityRegularizer/LogLog'dense_6/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes	
:?2!
dense_6/ActivityRegularizer/Log?
!dense_6/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_6/ActivityRegularizer/mul/x?
dense_6/ActivityRegularizer/mulMul*dense_6/ActivityRegularizer/mul/x:output:0#dense_6/ActivityRegularizer/Log:y:0*
T0*
_output_shapes	
:?2!
dense_6/ActivityRegularizer/mul?
!dense_6/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!dense_6/ActivityRegularizer/sub/x?
dense_6/ActivityRegularizer/subSub*dense_6/ActivityRegularizer/sub/x:output:0'dense_6/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2!
dense_6/ActivityRegularizer/sub?
'dense_6/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2)
'dense_6/ActivityRegularizer/truediv_1/x?
%dense_6/ActivityRegularizer/truediv_1RealDiv0dense_6/ActivityRegularizer/truediv_1/x:output:0#dense_6/ActivityRegularizer/sub:z:0*
T0*
_output_shapes	
:?2'
%dense_6/ActivityRegularizer/truediv_1?
!dense_6/ActivityRegularizer/Log_1Log)dense_6/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes	
:?2#
!dense_6/ActivityRegularizer/Log_1?
#dense_6/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2%
#dense_6/ActivityRegularizer/mul_1/x?
!dense_6/ActivityRegularizer/mul_1Mul,dense_6/ActivityRegularizer/mul_1/x:output:0%dense_6/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes	
:?2#
!dense_6/ActivityRegularizer/mul_1?
dense_6/ActivityRegularizer/addAddV2#dense_6/ActivityRegularizer/mul:z:0%dense_6/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes	
:?2!
dense_6/ActivityRegularizer/add?
!dense_6/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_6/ActivityRegularizer/Const?
dense_6/ActivityRegularizer/SumSum#dense_6/ActivityRegularizer/add:z:0*dense_6/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_6/ActivityRegularizer/Sum?
#dense_6/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#dense_6/ActivityRegularizer/mul_2/x?
!dense_6/ActivityRegularizer/mul_2Mul,dense_6/ActivityRegularizer/mul_2/x:output:0(dense_6/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_6/ActivityRegularizer/mul_2?
!dense_6/ActivityRegularizer/ShapeShapedense_6/Sigmoid:y:0*
T0*
_output_shapes
:2#
!dense_6/ActivityRegularizer/Shape?
/dense_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_6/ActivityRegularizer/strided_slice/stack?
1dense_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_1?
1dense_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_2?
)dense_6/ActivityRegularizer/strided_sliceStridedSlice*dense_6/ActivityRegularizer/Shape:output:08dense_6/ActivityRegularizer/strided_slice/stack:output:0:dense_6/ActivityRegularizer/strided_slice/stack_1:output:0:dense_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_6/ActivityRegularizer/strided_slice?
 dense_6/ActivityRegularizer/CastCast2dense_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_6/ActivityRegularizer/Cast?
%dense_6/ActivityRegularizer/truediv_2RealDiv%dense_6/ActivityRegularizer/mul_2:z:0$dense_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_6/ActivityRegularizer/truediv_2?
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOp?
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_6/kernel/Regularizer/Square?
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/Const?
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/Sum?
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_6/kernel/Regularizer/mul/x?
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mul?
IdentityIdentitydense_6/Sigmoid:y:0^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity)dense_6/ActivityRegularizer/truediv_2:z:0^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
0__inference_autoencoder_3_layer_call_fn_14423450
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
K__inference_autoencoder_3_layer_call_and_return_conditional_losses_144231872
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
?
?
K__inference_autoencoder_3_layer_call_and_return_conditional_losses_14423237
x)
sequential_6_14423218:
??$
sequential_6_14423220:	?)
sequential_7_14423224:
??$
sequential_7_14423226:	?
identity

identity_1??0dense_6/kernel/Regularizer/Square/ReadVariableOp?$sequential_6/StatefulPartitionedCall?$sequential_7/StatefulPartitionedCall?
$sequential_6/StatefulPartitionedCallStatefulPartitionedCallxsequential_6_14423218sequential_6_14423220*
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_144229992&
$sequential_6/StatefulPartitionedCall?
$sequential_7/StatefulPartitionedCallStatefulPartitionedCall-sequential_6/StatefulPartitionedCall:output:0sequential_7_14423224sequential_7_14423226*
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
J__inference_sequential_7_layer_call_and_return_conditional_losses_144231272&
$sequential_7/StatefulPartitionedCall?
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_6_14423218* 
_output_shapes
:
??*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOp?
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_6/kernel/Regularizer/Square?
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/Const?
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/Sum?
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_6/kernel/Regularizer/mul/x?
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mul?
IdentityIdentity-sequential_7/StatefulPartitionedCall:output:01^dense_6/kernel/Regularizer/Square/ReadVariableOp%^sequential_6/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity-sequential_6/StatefulPartitionedCall:output:11^dense_6/kernel/Regularizer/Square/ReadVariableOp%^sequential_6/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall2L
$sequential_7/StatefulPartitionedCall$sequential_7/StatefulPartitionedCall:K G
(
_output_shapes
:??????????

_user_specified_nameX
?
?
__inference_loss_fn_0_14423815M
9dense_6_kernel_regularizer_square_readvariableop_resource:
??
identity??0dense_6/kernel/Regularizer/Square/ReadVariableOp?
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9dense_6_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOp?
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_6/kernel/Regularizer/Square?
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/Const?
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/Sum?
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_6/kernel/Regularizer/mul/x?
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mul?
IdentityIdentity"dense_6/kernel/Regularizer/mul:z:01^dense_6/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp
?
?
J__inference_sequential_7_layer_call_and_return_conditional_losses_14423709

inputs:
&dense_7_matmul_readvariableop_resource:
??6
'dense_7_biasadd_readvariableop_resource:	?
identity??dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMulinputs%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_7/BiasAddz
dense_7/SigmoidSigmoiddense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_7/Sigmoid?
IdentityIdentitydense_7/Sigmoid:y:0^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_dense_7_layer_call_and_return_conditional_losses_14423826

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
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
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_7_layer_call_fn_14423760

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
J__inference_sequential_7_layer_call_and_return_conditional_losses_144230902
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
/__inference_sequential_7_layer_call_fn_14423778
dense_7_input
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_7_inputunknown	unknown_0*
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
J__inference_sequential_7_layer_call_and_return_conditional_losses_144231272
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
_user_specified_namedense_7_input
?
?
E__inference_dense_6_layer_call_and_return_conditional_losses_14422911

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_6/kernel/Regularizer/Square/ReadVariableOp?
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
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOp?
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_6/kernel/Regularizer/Square?
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/Const?
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/Sum?
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_6/kernel/Regularizer/mul/x?
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp*
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
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_7_layer_call_and_return_conditional_losses_14423127

inputs$
dense_7_14423121:
??
dense_7_14423123:	?
identity??dense_7/StatefulPartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCallinputsdense_7_14423121dense_7_14423123*
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
E__inference_dense_7_layer_call_and_return_conditional_losses_144230832!
dense_7/StatefulPartitionedCall?
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_7/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
0__inference_autoencoder_3_layer_call_fn_14423263
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
K__inference_autoencoder_3_layer_call_and_return_conditional_losses_144232372
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
?
?
E__inference_dense_6_layer_call_and_return_conditional_losses_14423852

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?0dense_6/kernel/Regularizer/Square/ReadVariableOp?
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
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOp?
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_6/kernel/Regularizer/Square?
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/Const?
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/Sum?
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_6/kernel/Regularizer/mul/x?
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp*
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
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
K__inference_autoencoder_3_layer_call_and_return_conditional_losses_14423285
input_1)
sequential_6_14423266:
??$
sequential_6_14423268:	?)
sequential_7_14423272:
??$
sequential_7_14423274:	?
identity

identity_1??0dense_6/kernel/Regularizer/Square/ReadVariableOp?$sequential_6/StatefulPartitionedCall?$sequential_7/StatefulPartitionedCall?
$sequential_6/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_6_14423266sequential_6_14423268*
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_144229332&
$sequential_6/StatefulPartitionedCall?
$sequential_7/StatefulPartitionedCallStatefulPartitionedCall-sequential_6/StatefulPartitionedCall:output:0sequential_7_14423272sequential_7_14423274*
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
J__inference_sequential_7_layer_call_and_return_conditional_losses_144230902&
$sequential_7/StatefulPartitionedCall?
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_6_14423266* 
_output_shapes
:
??*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOp?
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_6/kernel/Regularizer/Square?
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/Const?
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/Sum?
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_6/kernel/Regularizer/mul/x?
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mul?
IdentityIdentity-sequential_7/StatefulPartitionedCall:output:01^dense_6/kernel/Regularizer/Square/ReadVariableOp%^sequential_6/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity-sequential_6/StatefulPartitionedCall:output:11^dense_6/kernel/Regularizer/Square/ReadVariableOp%^sequential_6/StatefulPartitionedCall%^sequential_7/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall2L
$sequential_7/StatefulPartitionedCall$sequential_7/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
/__inference_sequential_6_layer_call_fn_14423688

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
J__inference_sequential_6_layer_call_and_return_conditional_losses_144229992
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
?Z
?
K__inference_autoencoder_3_layer_call_and_return_conditional_losses_14423436
xG
3sequential_6_dense_6_matmul_readvariableop_resource:
??C
4sequential_6_dense_6_biasadd_readvariableop_resource:	?G
3sequential_7_dense_7_matmul_readvariableop_resource:
??C
4sequential_7_dense_7_biasadd_readvariableop_resource:	?
identity

identity_1??0dense_6/kernel/Regularizer/Square/ReadVariableOp?+sequential_6/dense_6/BiasAdd/ReadVariableOp?*sequential_6/dense_6/MatMul/ReadVariableOp?+sequential_7/dense_7/BiasAdd/ReadVariableOp?*sequential_7/dense_7/MatMul/ReadVariableOp?
*sequential_6/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_6_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential_6/dense_6/MatMul/ReadVariableOp?
sequential_6/dense_6/MatMulMatMulx2sequential_6/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_6/dense_6/MatMul?
+sequential_6/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_6_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_6/dense_6/BiasAdd/ReadVariableOp?
sequential_6/dense_6/BiasAddBiasAdd%sequential_6/dense_6/MatMul:product:03sequential_6/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_6/dense_6/BiasAdd?
sequential_6/dense_6/SigmoidSigmoid%sequential_6/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_6/dense_6/Sigmoid?
0sequential_6/dense_6/ActivityRegularizer/SigmoidSigmoid sequential_6/dense_6/Sigmoid:y:0*
T0*(
_output_shapes
:??????????22
0sequential_6/dense_6/ActivityRegularizer/Sigmoid?
?sequential_6/dense_6/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2A
?sequential_6/dense_6/ActivityRegularizer/Mean/reduction_indices?
-sequential_6/dense_6/ActivityRegularizer/MeanMean4sequential_6/dense_6/ActivityRegularizer/Sigmoid:y:0Hsequential_6/dense_6/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes	
:?2/
-sequential_6/dense_6/ActivityRegularizer/Mean?
2sequential_6/dense_6/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.24
2sequential_6/dense_6/ActivityRegularizer/Maximum/y?
0sequential_6/dense_6/ActivityRegularizer/MaximumMaximum6sequential_6/dense_6/ActivityRegularizer/Mean:output:0;sequential_6/dense_6/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes	
:?22
0sequential_6/dense_6/ActivityRegularizer/Maximum?
2sequential_6/dense_6/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<24
2sequential_6/dense_6/ActivityRegularizer/truediv/x?
0sequential_6/dense_6/ActivityRegularizer/truedivRealDiv;sequential_6/dense_6/ActivityRegularizer/truediv/x:output:04sequential_6/dense_6/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?22
0sequential_6/dense_6/ActivityRegularizer/truediv?
,sequential_6/dense_6/ActivityRegularizer/LogLog4sequential_6/dense_6/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes	
:?2.
,sequential_6/dense_6/ActivityRegularizer/Log?
.sequential_6/dense_6/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.sequential_6/dense_6/ActivityRegularizer/mul/x?
,sequential_6/dense_6/ActivityRegularizer/mulMul7sequential_6/dense_6/ActivityRegularizer/mul/x:output:00sequential_6/dense_6/ActivityRegularizer/Log:y:0*
T0*
_output_shapes	
:?2.
,sequential_6/dense_6/ActivityRegularizer/mul?
.sequential_6/dense_6/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??20
.sequential_6/dense_6/ActivityRegularizer/sub/x?
,sequential_6/dense_6/ActivityRegularizer/subSub7sequential_6/dense_6/ActivityRegularizer/sub/x:output:04sequential_6/dense_6/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2.
,sequential_6/dense_6/ActivityRegularizer/sub?
4sequential_6/dense_6/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?26
4sequential_6/dense_6/ActivityRegularizer/truediv_1/x?
2sequential_6/dense_6/ActivityRegularizer/truediv_1RealDiv=sequential_6/dense_6/ActivityRegularizer/truediv_1/x:output:00sequential_6/dense_6/ActivityRegularizer/sub:z:0*
T0*
_output_shapes	
:?24
2sequential_6/dense_6/ActivityRegularizer/truediv_1?
.sequential_6/dense_6/ActivityRegularizer/Log_1Log6sequential_6/dense_6/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes	
:?20
.sequential_6/dense_6/ActivityRegularizer/Log_1?
0sequential_6/dense_6/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?22
0sequential_6/dense_6/ActivityRegularizer/mul_1/x?
.sequential_6/dense_6/ActivityRegularizer/mul_1Mul9sequential_6/dense_6/ActivityRegularizer/mul_1/x:output:02sequential_6/dense_6/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes	
:?20
.sequential_6/dense_6/ActivityRegularizer/mul_1?
,sequential_6/dense_6/ActivityRegularizer/addAddV20sequential_6/dense_6/ActivityRegularizer/mul:z:02sequential_6/dense_6/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes	
:?2.
,sequential_6/dense_6/ActivityRegularizer/add?
.sequential_6/dense_6/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_6/dense_6/ActivityRegularizer/Const?
,sequential_6/dense_6/ActivityRegularizer/SumSum0sequential_6/dense_6/ActivityRegularizer/add:z:07sequential_6/dense_6/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2.
,sequential_6/dense_6/ActivityRegularizer/Sum?
0sequential_6/dense_6/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??22
0sequential_6/dense_6/ActivityRegularizer/mul_2/x?
.sequential_6/dense_6/ActivityRegularizer/mul_2Mul9sequential_6/dense_6/ActivityRegularizer/mul_2/x:output:05sequential_6/dense_6/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 20
.sequential_6/dense_6/ActivityRegularizer/mul_2?
.sequential_6/dense_6/ActivityRegularizer/ShapeShape sequential_6/dense_6/Sigmoid:y:0*
T0*
_output_shapes
:20
.sequential_6/dense_6/ActivityRegularizer/Shape?
<sequential_6/dense_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential_6/dense_6/ActivityRegularizer/strided_slice/stack?
>sequential_6/dense_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_6/dense_6/ActivityRegularizer/strided_slice/stack_1?
>sequential_6/dense_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_6/dense_6/ActivityRegularizer/strided_slice/stack_2?
6sequential_6/dense_6/ActivityRegularizer/strided_sliceStridedSlice7sequential_6/dense_6/ActivityRegularizer/Shape:output:0Esequential_6/dense_6/ActivityRegularizer/strided_slice/stack:output:0Gsequential_6/dense_6/ActivityRegularizer/strided_slice/stack_1:output:0Gsequential_6/dense_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6sequential_6/dense_6/ActivityRegularizer/strided_slice?
-sequential_6/dense_6/ActivityRegularizer/CastCast?sequential_6/dense_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-sequential_6/dense_6/ActivityRegularizer/Cast?
2sequential_6/dense_6/ActivityRegularizer/truediv_2RealDiv2sequential_6/dense_6/ActivityRegularizer/mul_2:z:01sequential_6/dense_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 24
2sequential_6/dense_6/ActivityRegularizer/truediv_2?
*sequential_7/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_7_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential_7/dense_7/MatMul/ReadVariableOp?
sequential_7/dense_7/MatMulMatMul sequential_6/dense_6/Sigmoid:y:02sequential_7/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_7/dense_7/MatMul?
+sequential_7/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_7_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_7/dense_7/BiasAdd/ReadVariableOp?
sequential_7/dense_7/BiasAddBiasAdd%sequential_7/dense_7/MatMul:product:03sequential_7/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_7/dense_7/BiasAdd?
sequential_7/dense_7/SigmoidSigmoid%sequential_7/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_7/dense_7/Sigmoid?
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_6_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOp?
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_6/kernel/Regularizer/Square?
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/Const?
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/Sum?
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_6/kernel/Regularizer/mul/x?
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mul?
IdentityIdentity sequential_7/dense_7/Sigmoid:y:01^dense_6/kernel/Regularizer/Square/ReadVariableOp,^sequential_6/dense_6/BiasAdd/ReadVariableOp+^sequential_6/dense_6/MatMul/ReadVariableOp,^sequential_7/dense_7/BiasAdd/ReadVariableOp+^sequential_7/dense_7/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity6sequential_6/dense_6/ActivityRegularizer/truediv_2:z:01^dense_6/kernel/Regularizer/Square/ReadVariableOp,^sequential_6/dense_6/BiasAdd/ReadVariableOp+^sequential_6/dense_6/MatMul/ReadVariableOp,^sequential_7/dense_7/BiasAdd/ReadVariableOp+^sequential_7/dense_7/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp2Z
+sequential_6/dense_6/BiasAdd/ReadVariableOp+sequential_6/dense_6/BiasAdd/ReadVariableOp2X
*sequential_6/dense_6/MatMul/ReadVariableOp*sequential_6/dense_6/MatMul/ReadVariableOp2Z
+sequential_7/dense_7/BiasAdd/ReadVariableOp+sequential_7/dense_7/BiasAdd/ReadVariableOp2X
*sequential_7/dense_7/MatMul/ReadVariableOp*sequential_7/dense_7/MatMul/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_nameX
?A
?
J__inference_sequential_6_layer_call_and_return_conditional_losses_14423564

inputs:
&dense_6_matmul_readvariableop_resource:
??6
'dense_6_biasadd_readvariableop_resource:	?
identity

identity_1??dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?0dense_6/kernel/Regularizer/Square/ReadVariableOp?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMulinputs%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/BiasAddz
dense_6/SigmoidSigmoiddense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_6/Sigmoid?
#dense_6/ActivityRegularizer/SigmoidSigmoiddense_6/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2%
#dense_6/ActivityRegularizer/Sigmoid?
2dense_6/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 24
2dense_6/ActivityRegularizer/Mean/reduction_indices?
 dense_6/ActivityRegularizer/MeanMean'dense_6/ActivityRegularizer/Sigmoid:y:0;dense_6/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes	
:?2"
 dense_6/ActivityRegularizer/Mean?
%dense_6/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2'
%dense_6/ActivityRegularizer/Maximum/y?
#dense_6/ActivityRegularizer/MaximumMaximum)dense_6/ActivityRegularizer/Mean:output:0.dense_6/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes	
:?2%
#dense_6/ActivityRegularizer/Maximum?
%dense_6/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2'
%dense_6/ActivityRegularizer/truediv/x?
#dense_6/ActivityRegularizer/truedivRealDiv.dense_6/ActivityRegularizer/truediv/x:output:0'dense_6/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2%
#dense_6/ActivityRegularizer/truediv?
dense_6/ActivityRegularizer/LogLog'dense_6/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes	
:?2!
dense_6/ActivityRegularizer/Log?
!dense_6/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_6/ActivityRegularizer/mul/x?
dense_6/ActivityRegularizer/mulMul*dense_6/ActivityRegularizer/mul/x:output:0#dense_6/ActivityRegularizer/Log:y:0*
T0*
_output_shapes	
:?2!
dense_6/ActivityRegularizer/mul?
!dense_6/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!dense_6/ActivityRegularizer/sub/x?
dense_6/ActivityRegularizer/subSub*dense_6/ActivityRegularizer/sub/x:output:0'dense_6/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2!
dense_6/ActivityRegularizer/sub?
'dense_6/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2)
'dense_6/ActivityRegularizer/truediv_1/x?
%dense_6/ActivityRegularizer/truediv_1RealDiv0dense_6/ActivityRegularizer/truediv_1/x:output:0#dense_6/ActivityRegularizer/sub:z:0*
T0*
_output_shapes	
:?2'
%dense_6/ActivityRegularizer/truediv_1?
!dense_6/ActivityRegularizer/Log_1Log)dense_6/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes	
:?2#
!dense_6/ActivityRegularizer/Log_1?
#dense_6/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2%
#dense_6/ActivityRegularizer/mul_1/x?
!dense_6/ActivityRegularizer/mul_1Mul,dense_6/ActivityRegularizer/mul_1/x:output:0%dense_6/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes	
:?2#
!dense_6/ActivityRegularizer/mul_1?
dense_6/ActivityRegularizer/addAddV2#dense_6/ActivityRegularizer/mul:z:0%dense_6/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes	
:?2!
dense_6/ActivityRegularizer/add?
!dense_6/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_6/ActivityRegularizer/Const?
dense_6/ActivityRegularizer/SumSum#dense_6/ActivityRegularizer/add:z:0*dense_6/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_6/ActivityRegularizer/Sum?
#dense_6/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#dense_6/ActivityRegularizer/mul_2/x?
!dense_6/ActivityRegularizer/mul_2Mul,dense_6/ActivityRegularizer/mul_2/x:output:0(dense_6/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_6/ActivityRegularizer/mul_2?
!dense_6/ActivityRegularizer/ShapeShapedense_6/Sigmoid:y:0*
T0*
_output_shapes
:2#
!dense_6/ActivityRegularizer/Shape?
/dense_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_6/ActivityRegularizer/strided_slice/stack?
1dense_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_1?
1dense_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_2?
)dense_6/ActivityRegularizer/strided_sliceStridedSlice*dense_6/ActivityRegularizer/Shape:output:08dense_6/ActivityRegularizer/strided_slice/stack:output:0:dense_6/ActivityRegularizer/strided_slice/stack_1:output:0:dense_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_6/ActivityRegularizer/strided_slice?
 dense_6/ActivityRegularizer/CastCast2dense_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_6/ActivityRegularizer/Cast?
%dense_6/ActivityRegularizer/truediv_2RealDiv%dense_6/ActivityRegularizer/mul_2:z:0$dense_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_6/ActivityRegularizer/truediv_2?
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOp?
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_6/kernel/Regularizer/Square?
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/Const?
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/Sum?
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_6/kernel/Regularizer/mul/x?
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mul?
IdentityIdentitydense_6/Sigmoid:y:0^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity)dense_6/ActivityRegularizer/truediv_2:z:0^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_6_layer_call_fn_14423668
dense_6_input
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_6_inputunknown	unknown_0*
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_144229332
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
StatefulPartitionedCallStatefulPartitionedCall:W S
(
_output_shapes
:??????????
'
_user_specified_namedense_6_input
?
?
/__inference_sequential_6_layer_call_fn_14423678

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
J__inference_sequential_6_layer_call_and_return_conditional_losses_144229332
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
?B
?
J__inference_sequential_6_layer_call_and_return_conditional_losses_14423611
dense_6_input:
&dense_6_matmul_readvariableop_resource:
??6
'dense_6_biasadd_readvariableop_resource:	?
identity

identity_1??dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?0dense_6/kernel/Regularizer/Square/ReadVariableOp?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMuldense_6_input%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/BiasAddz
dense_6/SigmoidSigmoiddense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_6/Sigmoid?
#dense_6/ActivityRegularizer/SigmoidSigmoiddense_6/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2%
#dense_6/ActivityRegularizer/Sigmoid?
2dense_6/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 24
2dense_6/ActivityRegularizer/Mean/reduction_indices?
 dense_6/ActivityRegularizer/MeanMean'dense_6/ActivityRegularizer/Sigmoid:y:0;dense_6/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes	
:?2"
 dense_6/ActivityRegularizer/Mean?
%dense_6/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2'
%dense_6/ActivityRegularizer/Maximum/y?
#dense_6/ActivityRegularizer/MaximumMaximum)dense_6/ActivityRegularizer/Mean:output:0.dense_6/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes	
:?2%
#dense_6/ActivityRegularizer/Maximum?
%dense_6/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2'
%dense_6/ActivityRegularizer/truediv/x?
#dense_6/ActivityRegularizer/truedivRealDiv.dense_6/ActivityRegularizer/truediv/x:output:0'dense_6/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2%
#dense_6/ActivityRegularizer/truediv?
dense_6/ActivityRegularizer/LogLog'dense_6/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes	
:?2!
dense_6/ActivityRegularizer/Log?
!dense_6/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_6/ActivityRegularizer/mul/x?
dense_6/ActivityRegularizer/mulMul*dense_6/ActivityRegularizer/mul/x:output:0#dense_6/ActivityRegularizer/Log:y:0*
T0*
_output_shapes	
:?2!
dense_6/ActivityRegularizer/mul?
!dense_6/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!dense_6/ActivityRegularizer/sub/x?
dense_6/ActivityRegularizer/subSub*dense_6/ActivityRegularizer/sub/x:output:0'dense_6/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2!
dense_6/ActivityRegularizer/sub?
'dense_6/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2)
'dense_6/ActivityRegularizer/truediv_1/x?
%dense_6/ActivityRegularizer/truediv_1RealDiv0dense_6/ActivityRegularizer/truediv_1/x:output:0#dense_6/ActivityRegularizer/sub:z:0*
T0*
_output_shapes	
:?2'
%dense_6/ActivityRegularizer/truediv_1?
!dense_6/ActivityRegularizer/Log_1Log)dense_6/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes	
:?2#
!dense_6/ActivityRegularizer/Log_1?
#dense_6/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2%
#dense_6/ActivityRegularizer/mul_1/x?
!dense_6/ActivityRegularizer/mul_1Mul,dense_6/ActivityRegularizer/mul_1/x:output:0%dense_6/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes	
:?2#
!dense_6/ActivityRegularizer/mul_1?
dense_6/ActivityRegularizer/addAddV2#dense_6/ActivityRegularizer/mul:z:0%dense_6/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes	
:?2!
dense_6/ActivityRegularizer/add?
!dense_6/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_6/ActivityRegularizer/Const?
dense_6/ActivityRegularizer/SumSum#dense_6/ActivityRegularizer/add:z:0*dense_6/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_6/ActivityRegularizer/Sum?
#dense_6/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#dense_6/ActivityRegularizer/mul_2/x?
!dense_6/ActivityRegularizer/mul_2Mul,dense_6/ActivityRegularizer/mul_2/x:output:0(dense_6/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_6/ActivityRegularizer/mul_2?
!dense_6/ActivityRegularizer/ShapeShapedense_6/Sigmoid:y:0*
T0*
_output_shapes
:2#
!dense_6/ActivityRegularizer/Shape?
/dense_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_6/ActivityRegularizer/strided_slice/stack?
1dense_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_1?
1dense_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_2?
)dense_6/ActivityRegularizer/strided_sliceStridedSlice*dense_6/ActivityRegularizer/Shape:output:08dense_6/ActivityRegularizer/strided_slice/stack:output:0:dense_6/ActivityRegularizer/strided_slice/stack_1:output:0:dense_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_6/ActivityRegularizer/strided_slice?
 dense_6/ActivityRegularizer/CastCast2dense_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_6/ActivityRegularizer/Cast?
%dense_6/ActivityRegularizer/truediv_2RealDiv%dense_6/ActivityRegularizer/mul_2:z:0$dense_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_6/ActivityRegularizer/truediv_2?
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOp?
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_6/kernel/Regularizer/Square?
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/Const?
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/Sum?
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_6/kernel/Regularizer/mul/x?
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mul?
IdentityIdentitydense_6/Sigmoid:y:0^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity)dense_6/ActivityRegularizer/truediv_2:z:0^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp:W S
(
_output_shapes
:??????????
'
_user_specified_namedense_6_input
?Z
?
K__inference_autoencoder_3_layer_call_and_return_conditional_losses_14423382
xG
3sequential_6_dense_6_matmul_readvariableop_resource:
??C
4sequential_6_dense_6_biasadd_readvariableop_resource:	?G
3sequential_7_dense_7_matmul_readvariableop_resource:
??C
4sequential_7_dense_7_biasadd_readvariableop_resource:	?
identity

identity_1??0dense_6/kernel/Regularizer/Square/ReadVariableOp?+sequential_6/dense_6/BiasAdd/ReadVariableOp?*sequential_6/dense_6/MatMul/ReadVariableOp?+sequential_7/dense_7/BiasAdd/ReadVariableOp?*sequential_7/dense_7/MatMul/ReadVariableOp?
*sequential_6/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_6_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential_6/dense_6/MatMul/ReadVariableOp?
sequential_6/dense_6/MatMulMatMulx2sequential_6/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_6/dense_6/MatMul?
+sequential_6/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_6_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_6/dense_6/BiasAdd/ReadVariableOp?
sequential_6/dense_6/BiasAddBiasAdd%sequential_6/dense_6/MatMul:product:03sequential_6/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_6/dense_6/BiasAdd?
sequential_6/dense_6/SigmoidSigmoid%sequential_6/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_6/dense_6/Sigmoid?
0sequential_6/dense_6/ActivityRegularizer/SigmoidSigmoid sequential_6/dense_6/Sigmoid:y:0*
T0*(
_output_shapes
:??????????22
0sequential_6/dense_6/ActivityRegularizer/Sigmoid?
?sequential_6/dense_6/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2A
?sequential_6/dense_6/ActivityRegularizer/Mean/reduction_indices?
-sequential_6/dense_6/ActivityRegularizer/MeanMean4sequential_6/dense_6/ActivityRegularizer/Sigmoid:y:0Hsequential_6/dense_6/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes	
:?2/
-sequential_6/dense_6/ActivityRegularizer/Mean?
2sequential_6/dense_6/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.24
2sequential_6/dense_6/ActivityRegularizer/Maximum/y?
0sequential_6/dense_6/ActivityRegularizer/MaximumMaximum6sequential_6/dense_6/ActivityRegularizer/Mean:output:0;sequential_6/dense_6/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes	
:?22
0sequential_6/dense_6/ActivityRegularizer/Maximum?
2sequential_6/dense_6/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<24
2sequential_6/dense_6/ActivityRegularizer/truediv/x?
0sequential_6/dense_6/ActivityRegularizer/truedivRealDiv;sequential_6/dense_6/ActivityRegularizer/truediv/x:output:04sequential_6/dense_6/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?22
0sequential_6/dense_6/ActivityRegularizer/truediv?
,sequential_6/dense_6/ActivityRegularizer/LogLog4sequential_6/dense_6/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes	
:?2.
,sequential_6/dense_6/ActivityRegularizer/Log?
.sequential_6/dense_6/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<20
.sequential_6/dense_6/ActivityRegularizer/mul/x?
,sequential_6/dense_6/ActivityRegularizer/mulMul7sequential_6/dense_6/ActivityRegularizer/mul/x:output:00sequential_6/dense_6/ActivityRegularizer/Log:y:0*
T0*
_output_shapes	
:?2.
,sequential_6/dense_6/ActivityRegularizer/mul?
.sequential_6/dense_6/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??20
.sequential_6/dense_6/ActivityRegularizer/sub/x?
,sequential_6/dense_6/ActivityRegularizer/subSub7sequential_6/dense_6/ActivityRegularizer/sub/x:output:04sequential_6/dense_6/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2.
,sequential_6/dense_6/ActivityRegularizer/sub?
4sequential_6/dense_6/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?26
4sequential_6/dense_6/ActivityRegularizer/truediv_1/x?
2sequential_6/dense_6/ActivityRegularizer/truediv_1RealDiv=sequential_6/dense_6/ActivityRegularizer/truediv_1/x:output:00sequential_6/dense_6/ActivityRegularizer/sub:z:0*
T0*
_output_shapes	
:?24
2sequential_6/dense_6/ActivityRegularizer/truediv_1?
.sequential_6/dense_6/ActivityRegularizer/Log_1Log6sequential_6/dense_6/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes	
:?20
.sequential_6/dense_6/ActivityRegularizer/Log_1?
0sequential_6/dense_6/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?22
0sequential_6/dense_6/ActivityRegularizer/mul_1/x?
.sequential_6/dense_6/ActivityRegularizer/mul_1Mul9sequential_6/dense_6/ActivityRegularizer/mul_1/x:output:02sequential_6/dense_6/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes	
:?20
.sequential_6/dense_6/ActivityRegularizer/mul_1?
,sequential_6/dense_6/ActivityRegularizer/addAddV20sequential_6/dense_6/ActivityRegularizer/mul:z:02sequential_6/dense_6/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes	
:?2.
,sequential_6/dense_6/ActivityRegularizer/add?
.sequential_6/dense_6/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential_6/dense_6/ActivityRegularizer/Const?
,sequential_6/dense_6/ActivityRegularizer/SumSum0sequential_6/dense_6/ActivityRegularizer/add:z:07sequential_6/dense_6/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2.
,sequential_6/dense_6/ActivityRegularizer/Sum?
0sequential_6/dense_6/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??22
0sequential_6/dense_6/ActivityRegularizer/mul_2/x?
.sequential_6/dense_6/ActivityRegularizer/mul_2Mul9sequential_6/dense_6/ActivityRegularizer/mul_2/x:output:05sequential_6/dense_6/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 20
.sequential_6/dense_6/ActivityRegularizer/mul_2?
.sequential_6/dense_6/ActivityRegularizer/ShapeShape sequential_6/dense_6/Sigmoid:y:0*
T0*
_output_shapes
:20
.sequential_6/dense_6/ActivityRegularizer/Shape?
<sequential_6/dense_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential_6/dense_6/ActivityRegularizer/strided_slice/stack?
>sequential_6/dense_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_6/dense_6/ActivityRegularizer/strided_slice/stack_1?
>sequential_6/dense_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_6/dense_6/ActivityRegularizer/strided_slice/stack_2?
6sequential_6/dense_6/ActivityRegularizer/strided_sliceStridedSlice7sequential_6/dense_6/ActivityRegularizer/Shape:output:0Esequential_6/dense_6/ActivityRegularizer/strided_slice/stack:output:0Gsequential_6/dense_6/ActivityRegularizer/strided_slice/stack_1:output:0Gsequential_6/dense_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6sequential_6/dense_6/ActivityRegularizer/strided_slice?
-sequential_6/dense_6/ActivityRegularizer/CastCast?sequential_6/dense_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-sequential_6/dense_6/ActivityRegularizer/Cast?
2sequential_6/dense_6/ActivityRegularizer/truediv_2RealDiv2sequential_6/dense_6/ActivityRegularizer/mul_2:z:01sequential_6/dense_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 24
2sequential_6/dense_6/ActivityRegularizer/truediv_2?
*sequential_7/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_7_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*sequential_7/dense_7/MatMul/ReadVariableOp?
sequential_7/dense_7/MatMulMatMul sequential_6/dense_6/Sigmoid:y:02sequential_7/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_7/dense_7/MatMul?
+sequential_7/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_7_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+sequential_7/dense_7/BiasAdd/ReadVariableOp?
sequential_7/dense_7/BiasAddBiasAdd%sequential_7/dense_7/MatMul:product:03sequential_7/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_7/dense_7/BiasAdd?
sequential_7/dense_7/SigmoidSigmoid%sequential_7/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_7/dense_7/Sigmoid?
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp3sequential_6_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOp?
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_6/kernel/Regularizer/Square?
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/Const?
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/Sum?
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_6/kernel/Regularizer/mul/x?
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mul?
IdentityIdentity sequential_7/dense_7/Sigmoid:y:01^dense_6/kernel/Regularizer/Square/ReadVariableOp,^sequential_6/dense_6/BiasAdd/ReadVariableOp+^sequential_6/dense_6/MatMul/ReadVariableOp,^sequential_7/dense_7/BiasAdd/ReadVariableOp+^sequential_7/dense_7/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity6sequential_6/dense_6/ActivityRegularizer/truediv_2:z:01^dense_6/kernel/Regularizer/Square/ReadVariableOp,^sequential_6/dense_6/BiasAdd/ReadVariableOp+^sequential_6/dense_6/MatMul/ReadVariableOp,^sequential_7/dense_7/BiasAdd/ReadVariableOp+^sequential_7/dense_7/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp2Z
+sequential_6/dense_6/BiasAdd/ReadVariableOp+sequential_6/dense_6/BiasAdd/ReadVariableOp2X
*sequential_6/dense_6/MatMul/ReadVariableOp*sequential_6/dense_6/MatMul/ReadVariableOp2Z
+sequential_7/dense_7/BiasAdd/ReadVariableOp+sequential_7/dense_7/BiasAdd/ReadVariableOp2X
*sequential_7/dense_7/MatMul/ReadVariableOp*sequential_7/dense_7/MatMul/ReadVariableOp:K G
(
_output_shapes
:??????????

_user_specified_nameX
?
?
I__inference_dense_6_layer_call_and_return_all_conditional_losses_14423795

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
E__inference_dense_6_layer_call_and_return_conditional_losses_144229112
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
1__inference_dense_6_activity_regularizer_144228872
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
/__inference_sequential_7_layer_call_fn_14423751
dense_7_input
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_7_inputunknown	unknown_0*
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
J__inference_sequential_7_layer_call_and_return_conditional_losses_144230902
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
_user_specified_namedense_7_input
?
?
0__inference_autoencoder_3_layer_call_fn_14423199
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
K__inference_autoencoder_3_layer_call_and_return_conditional_losses_144231872
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
*__inference_dense_6_layer_call_fn_14423804

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
E__inference_dense_6_layer_call_and_return_conditional_losses_144229112
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
&__inference_signature_wrapper_14423328
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
#__inference__wrapped_model_144228572
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
?\
?
#__inference__wrapped_model_14422857
input_1U
Aautoencoder_3_sequential_6_dense_6_matmul_readvariableop_resource:
??Q
Bautoencoder_3_sequential_6_dense_6_biasadd_readvariableop_resource:	?U
Aautoencoder_3_sequential_7_dense_7_matmul_readvariableop_resource:
??Q
Bautoencoder_3_sequential_7_dense_7_biasadd_readvariableop_resource:	?
identity??9autoencoder_3/sequential_6/dense_6/BiasAdd/ReadVariableOp?8autoencoder_3/sequential_6/dense_6/MatMul/ReadVariableOp?9autoencoder_3/sequential_7/dense_7/BiasAdd/ReadVariableOp?8autoencoder_3/sequential_7/dense_7/MatMul/ReadVariableOp?
8autoencoder_3/sequential_6/dense_6/MatMul/ReadVariableOpReadVariableOpAautoencoder_3_sequential_6_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02:
8autoencoder_3/sequential_6/dense_6/MatMul/ReadVariableOp?
)autoencoder_3/sequential_6/dense_6/MatMulMatMulinput_1@autoencoder_3/sequential_6/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)autoencoder_3/sequential_6/dense_6/MatMul?
9autoencoder_3/sequential_6/dense_6/BiasAdd/ReadVariableOpReadVariableOpBautoencoder_3_sequential_6_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02;
9autoencoder_3/sequential_6/dense_6/BiasAdd/ReadVariableOp?
*autoencoder_3/sequential_6/dense_6/BiasAddBiasAdd3autoencoder_3/sequential_6/dense_6/MatMul:product:0Aautoencoder_3/sequential_6/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*autoencoder_3/sequential_6/dense_6/BiasAdd?
*autoencoder_3/sequential_6/dense_6/SigmoidSigmoid3autoencoder_3/sequential_6/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2,
*autoencoder_3/sequential_6/dense_6/Sigmoid?
>autoencoder_3/sequential_6/dense_6/ActivityRegularizer/SigmoidSigmoid.autoencoder_3/sequential_6/dense_6/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2@
>autoencoder_3/sequential_6/dense_6/ActivityRegularizer/Sigmoid?
Mautoencoder_3/sequential_6/dense_6/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mautoencoder_3/sequential_6/dense_6/ActivityRegularizer/Mean/reduction_indices?
;autoencoder_3/sequential_6/dense_6/ActivityRegularizer/MeanMeanBautoencoder_3/sequential_6/dense_6/ActivityRegularizer/Sigmoid:y:0Vautoencoder_3/sequential_6/dense_6/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes	
:?2=
;autoencoder_3/sequential_6/dense_6/ActivityRegularizer/Mean?
@autoencoder_3/sequential_6/dense_6/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2B
@autoencoder_3/sequential_6/dense_6/ActivityRegularizer/Maximum/y?
>autoencoder_3/sequential_6/dense_6/ActivityRegularizer/MaximumMaximumDautoencoder_3/sequential_6/dense_6/ActivityRegularizer/Mean:output:0Iautoencoder_3/sequential_6/dense_6/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes	
:?2@
>autoencoder_3/sequential_6/dense_6/ActivityRegularizer/Maximum?
@autoencoder_3/sequential_6/dense_6/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2B
@autoencoder_3/sequential_6/dense_6/ActivityRegularizer/truediv/x?
>autoencoder_3/sequential_6/dense_6/ActivityRegularizer/truedivRealDivIautoencoder_3/sequential_6/dense_6/ActivityRegularizer/truediv/x:output:0Bautoencoder_3/sequential_6/dense_6/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2@
>autoencoder_3/sequential_6/dense_6/ActivityRegularizer/truediv?
:autoencoder_3/sequential_6/dense_6/ActivityRegularizer/LogLogBautoencoder_3/sequential_6/dense_6/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes	
:?2<
:autoencoder_3/sequential_6/dense_6/ActivityRegularizer/Log?
<autoencoder_3/sequential_6/dense_6/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2>
<autoencoder_3/sequential_6/dense_6/ActivityRegularizer/mul/x?
:autoencoder_3/sequential_6/dense_6/ActivityRegularizer/mulMulEautoencoder_3/sequential_6/dense_6/ActivityRegularizer/mul/x:output:0>autoencoder_3/sequential_6/dense_6/ActivityRegularizer/Log:y:0*
T0*
_output_shapes	
:?2<
:autoencoder_3/sequential_6/dense_6/ActivityRegularizer/mul?
<autoencoder_3/sequential_6/dense_6/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2>
<autoencoder_3/sequential_6/dense_6/ActivityRegularizer/sub/x?
:autoencoder_3/sequential_6/dense_6/ActivityRegularizer/subSubEautoencoder_3/sequential_6/dense_6/ActivityRegularizer/sub/x:output:0Bautoencoder_3/sequential_6/dense_6/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2<
:autoencoder_3/sequential_6/dense_6/ActivityRegularizer/sub?
Bautoencoder_3/sequential_6/dense_6/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2D
Bautoencoder_3/sequential_6/dense_6/ActivityRegularizer/truediv_1/x?
@autoencoder_3/sequential_6/dense_6/ActivityRegularizer/truediv_1RealDivKautoencoder_3/sequential_6/dense_6/ActivityRegularizer/truediv_1/x:output:0>autoencoder_3/sequential_6/dense_6/ActivityRegularizer/sub:z:0*
T0*
_output_shapes	
:?2B
@autoencoder_3/sequential_6/dense_6/ActivityRegularizer/truediv_1?
<autoencoder_3/sequential_6/dense_6/ActivityRegularizer/Log_1LogDautoencoder_3/sequential_6/dense_6/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes	
:?2>
<autoencoder_3/sequential_6/dense_6/ActivityRegularizer/Log_1?
>autoencoder_3/sequential_6/dense_6/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2@
>autoencoder_3/sequential_6/dense_6/ActivityRegularizer/mul_1/x?
<autoencoder_3/sequential_6/dense_6/ActivityRegularizer/mul_1MulGautoencoder_3/sequential_6/dense_6/ActivityRegularizer/mul_1/x:output:0@autoencoder_3/sequential_6/dense_6/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes	
:?2>
<autoencoder_3/sequential_6/dense_6/ActivityRegularizer/mul_1?
:autoencoder_3/sequential_6/dense_6/ActivityRegularizer/addAddV2>autoencoder_3/sequential_6/dense_6/ActivityRegularizer/mul:z:0@autoencoder_3/sequential_6/dense_6/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes	
:?2<
:autoencoder_3/sequential_6/dense_6/ActivityRegularizer/add?
<autoencoder_3/sequential_6/dense_6/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2>
<autoencoder_3/sequential_6/dense_6/ActivityRegularizer/Const?
:autoencoder_3/sequential_6/dense_6/ActivityRegularizer/SumSum>autoencoder_3/sequential_6/dense_6/ActivityRegularizer/add:z:0Eautoencoder_3/sequential_6/dense_6/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2<
:autoencoder_3/sequential_6/dense_6/ActivityRegularizer/Sum?
>autoencoder_3/sequential_6/dense_6/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2@
>autoencoder_3/sequential_6/dense_6/ActivityRegularizer/mul_2/x?
<autoencoder_3/sequential_6/dense_6/ActivityRegularizer/mul_2MulGautoencoder_3/sequential_6/dense_6/ActivityRegularizer/mul_2/x:output:0Cautoencoder_3/sequential_6/dense_6/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2>
<autoencoder_3/sequential_6/dense_6/ActivityRegularizer/mul_2?
<autoencoder_3/sequential_6/dense_6/ActivityRegularizer/ShapeShape.autoencoder_3/sequential_6/dense_6/Sigmoid:y:0*
T0*
_output_shapes
:2>
<autoencoder_3/sequential_6/dense_6/ActivityRegularizer/Shape?
Jautoencoder_3/sequential_6/dense_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2L
Jautoencoder_3/sequential_6/dense_6/ActivityRegularizer/strided_slice/stack?
Lautoencoder_3/sequential_6/dense_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2N
Lautoencoder_3/sequential_6/dense_6/ActivityRegularizer/strided_slice/stack_1?
Lautoencoder_3/sequential_6/dense_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2N
Lautoencoder_3/sequential_6/dense_6/ActivityRegularizer/strided_slice/stack_2?
Dautoencoder_3/sequential_6/dense_6/ActivityRegularizer/strided_sliceStridedSliceEautoencoder_3/sequential_6/dense_6/ActivityRegularizer/Shape:output:0Sautoencoder_3/sequential_6/dense_6/ActivityRegularizer/strided_slice/stack:output:0Uautoencoder_3/sequential_6/dense_6/ActivityRegularizer/strided_slice/stack_1:output:0Uautoencoder_3/sequential_6/dense_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2F
Dautoencoder_3/sequential_6/dense_6/ActivityRegularizer/strided_slice?
;autoencoder_3/sequential_6/dense_6/ActivityRegularizer/CastCastMautoencoder_3/sequential_6/dense_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2=
;autoencoder_3/sequential_6/dense_6/ActivityRegularizer/Cast?
@autoencoder_3/sequential_6/dense_6/ActivityRegularizer/truediv_2RealDiv@autoencoder_3/sequential_6/dense_6/ActivityRegularizer/mul_2:z:0?autoencoder_3/sequential_6/dense_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2B
@autoencoder_3/sequential_6/dense_6/ActivityRegularizer/truediv_2?
8autoencoder_3/sequential_7/dense_7/MatMul/ReadVariableOpReadVariableOpAautoencoder_3_sequential_7_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02:
8autoencoder_3/sequential_7/dense_7/MatMul/ReadVariableOp?
)autoencoder_3/sequential_7/dense_7/MatMulMatMul.autoencoder_3/sequential_6/dense_6/Sigmoid:y:0@autoencoder_3/sequential_7/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)autoencoder_3/sequential_7/dense_7/MatMul?
9autoencoder_3/sequential_7/dense_7/BiasAdd/ReadVariableOpReadVariableOpBautoencoder_3_sequential_7_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02;
9autoencoder_3/sequential_7/dense_7/BiasAdd/ReadVariableOp?
*autoencoder_3/sequential_7/dense_7/BiasAddBiasAdd3autoencoder_3/sequential_7/dense_7/MatMul:product:0Aautoencoder_3/sequential_7/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*autoencoder_3/sequential_7/dense_7/BiasAdd?
*autoencoder_3/sequential_7/dense_7/SigmoidSigmoid3autoencoder_3/sequential_7/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2,
*autoencoder_3/sequential_7/dense_7/Sigmoid?
IdentityIdentity.autoencoder_3/sequential_7/dense_7/Sigmoid:y:0:^autoencoder_3/sequential_6/dense_6/BiasAdd/ReadVariableOp9^autoencoder_3/sequential_6/dense_6/MatMul/ReadVariableOp:^autoencoder_3/sequential_7/dense_7/BiasAdd/ReadVariableOp9^autoencoder_3/sequential_7/dense_7/MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : : : 2v
9autoencoder_3/sequential_6/dense_6/BiasAdd/ReadVariableOp9autoencoder_3/sequential_6/dense_6/BiasAdd/ReadVariableOp2t
8autoencoder_3/sequential_6/dense_6/MatMul/ReadVariableOp8autoencoder_3/sequential_6/dense_6/MatMul/ReadVariableOp2v
9autoencoder_3/sequential_7/dense_7/BiasAdd/ReadVariableOp9autoencoder_3/sequential_7/dense_7/BiasAdd/ReadVariableOp2t
8autoencoder_3/sequential_7/dense_7/MatMul/ReadVariableOp8autoencoder_3/sequential_7/dense_7/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?B
?
J__inference_sequential_6_layer_call_and_return_conditional_losses_14423658
dense_6_input:
&dense_6_matmul_readvariableop_resource:
??6
'dense_6_biasadd_readvariableop_resource:	?
identity

identity_1??dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?0dense_6/kernel/Regularizer/Square/ReadVariableOp?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMuldense_6_input%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_6/BiasAddz
dense_6/SigmoidSigmoiddense_6/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_6/Sigmoid?
#dense_6/ActivityRegularizer/SigmoidSigmoiddense_6/Sigmoid:y:0*
T0*(
_output_shapes
:??????????2%
#dense_6/ActivityRegularizer/Sigmoid?
2dense_6/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 24
2dense_6/ActivityRegularizer/Mean/reduction_indices?
 dense_6/ActivityRegularizer/MeanMean'dense_6/ActivityRegularizer/Sigmoid:y:0;dense_6/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes	
:?2"
 dense_6/ActivityRegularizer/Mean?
%dense_6/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2'
%dense_6/ActivityRegularizer/Maximum/y?
#dense_6/ActivityRegularizer/MaximumMaximum)dense_6/ActivityRegularizer/Mean:output:0.dense_6/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes	
:?2%
#dense_6/ActivityRegularizer/Maximum?
%dense_6/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2'
%dense_6/ActivityRegularizer/truediv/x?
#dense_6/ActivityRegularizer/truedivRealDiv.dense_6/ActivityRegularizer/truediv/x:output:0'dense_6/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2%
#dense_6/ActivityRegularizer/truediv?
dense_6/ActivityRegularizer/LogLog'dense_6/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes	
:?2!
dense_6/ActivityRegularizer/Log?
!dense_6/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_6/ActivityRegularizer/mul/x?
dense_6/ActivityRegularizer/mulMul*dense_6/ActivityRegularizer/mul/x:output:0#dense_6/ActivityRegularizer/Log:y:0*
T0*
_output_shapes	
:?2!
dense_6/ActivityRegularizer/mul?
!dense_6/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!dense_6/ActivityRegularizer/sub/x?
dense_6/ActivityRegularizer/subSub*dense_6/ActivityRegularizer/sub/x:output:0'dense_6/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes	
:?2!
dense_6/ActivityRegularizer/sub?
'dense_6/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2)
'dense_6/ActivityRegularizer/truediv_1/x?
%dense_6/ActivityRegularizer/truediv_1RealDiv0dense_6/ActivityRegularizer/truediv_1/x:output:0#dense_6/ActivityRegularizer/sub:z:0*
T0*
_output_shapes	
:?2'
%dense_6/ActivityRegularizer/truediv_1?
!dense_6/ActivityRegularizer/Log_1Log)dense_6/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes	
:?2#
!dense_6/ActivityRegularizer/Log_1?
#dense_6/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2%
#dense_6/ActivityRegularizer/mul_1/x?
!dense_6/ActivityRegularizer/mul_1Mul,dense_6/ActivityRegularizer/mul_1/x:output:0%dense_6/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes	
:?2#
!dense_6/ActivityRegularizer/mul_1?
dense_6/ActivityRegularizer/addAddV2#dense_6/ActivityRegularizer/mul:z:0%dense_6/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes	
:?2!
dense_6/ActivityRegularizer/add?
!dense_6/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_6/ActivityRegularizer/Const?
dense_6/ActivityRegularizer/SumSum#dense_6/ActivityRegularizer/add:z:0*dense_6/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_6/ActivityRegularizer/Sum?
#dense_6/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#dense_6/ActivityRegularizer/mul_2/x?
!dense_6/ActivityRegularizer/mul_2Mul,dense_6/ActivityRegularizer/mul_2/x:output:0(dense_6/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!dense_6/ActivityRegularizer/mul_2?
!dense_6/ActivityRegularizer/ShapeShapedense_6/Sigmoid:y:0*
T0*
_output_shapes
:2#
!dense_6/ActivityRegularizer/Shape?
/dense_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_6/ActivityRegularizer/strided_slice/stack?
1dense_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_1?
1dense_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_2?
)dense_6/ActivityRegularizer/strided_sliceStridedSlice*dense_6/ActivityRegularizer/Shape:output:08dense_6/ActivityRegularizer/strided_slice/stack:output:0:dense_6/ActivityRegularizer/strided_slice/stack_1:output:0:dense_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_6/ActivityRegularizer/strided_slice?
 dense_6/ActivityRegularizer/CastCast2dense_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_6/ActivityRegularizer/Cast?
%dense_6/ActivityRegularizer/truediv_2RealDiv%dense_6/ActivityRegularizer/mul_2:z:0$dense_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_6/ActivityRegularizer/truediv_2?
0dense_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0dense_6/kernel/Regularizer/Square/ReadVariableOp?
!dense_6/kernel/Regularizer/SquareSquare8dense_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2#
!dense_6/kernel/Regularizer/Square?
 dense_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense_6/kernel/Regularizer/Const?
dense_6/kernel/Regularizer/SumSum%dense_6/kernel/Regularizer/Square:y:0)dense_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/Sum?
 dense_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_6/kernel/Regularizer/mul/x?
dense_6/kernel/Regularizer/mulMul)dense_6/kernel/Regularizer/mul/x:output:0'dense_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_6/kernel/Regularizer/mul?
IdentityIdentitydense_6/Sigmoid:y:0^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity)dense_6/ActivityRegularizer/truediv_2:z:0^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp1^dense_6/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2d
0dense_6/kernel/Regularizer/Square/ReadVariableOp0dense_6/kernel/Regularizer/Square/ReadVariableOp:W S
(
_output_shapes
:??????????
'
_user_specified_namedense_6_input"?L
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
*8&call_and_return_all_conditional_losses
9__call__
:_default_save_signature"?
_tf_keras_model?{"name": "autoencoder_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Autoencoder", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 512]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Autoencoder"}}
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
<__call__"?
_tf_keras_sequential?{"name": "sequential_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_6_input"}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 256, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}}, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 6}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 512]}, "float32", "dense_6_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_6_input"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 256, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}]}}}
?
layer_with_weights-0
layer-0
	variables
trainable_variables
regularization_losses
	keras_api
*=&call_and_return_all_conditional_losses
>__call__"?
_tf_keras_sequential?{"name": "sequential_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_7_input"}}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 512, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 12}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [512, 256]}, "float32", "dense_7_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_7_input"}, "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 512, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10}]}}}
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
layer_regularization_losses
metrics

layers
layer_metrics
trainable_variables
regularization_losses
non_trainable_variables
9__call__
:_default_save_signature
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

_tf_keras_layer?
{"name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 256, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 6}, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 512]}}
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
 layer_regularization_losses
!metrics

"layers
#layer_metrics
trainable_variables
regularization_losses
$non_trainable_variables
<__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
?

kernel
bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
*C&call_and_return_all_conditional_losses
D__call__"?
_tf_keras_layer?{"name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 512, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 12}, "build_input_shape": {"class_name": "TensorShape", "items": [512, 256]}}
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
)layer_regularization_losses
*metrics

+layers
,layer_metrics
trainable_variables
regularization_losses
-non_trainable_variables
>__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_6/kernel
:?2dense_6/bias
": 
??2dense_7/kernel
:?2dense_7/bias
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
trainable_variables
.layer_regularization_losses
/metrics
0layer_metrics

1layers
regularization_losses
2non_trainable_variables
A__call__
Eactivity_regularizer_fn
*@&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
	0"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
?
%	variables
&trainable_variables
3layer_regularization_losses
4metrics
5layer_metrics

6layers
'regularization_losses
7non_trainable_variables
D__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
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
trackable_dict_wrapper
 "
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
 "
trackable_list_wrapper
?2?
K__inference_autoencoder_3_layer_call_and_return_conditional_losses_14423382
K__inference_autoencoder_3_layer_call_and_return_conditional_losses_14423436
K__inference_autoencoder_3_layer_call_and_return_conditional_losses_14423285
K__inference_autoencoder_3_layer_call_and_return_conditional_losses_14423307?
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
?2?
0__inference_autoencoder_3_layer_call_fn_14423199
0__inference_autoencoder_3_layer_call_fn_14423450
0__inference_autoencoder_3_layer_call_fn_14423464
0__inference_autoencoder_3_layer_call_fn_14423263?
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
#__inference__wrapped_model_14422857?
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_14423517
J__inference_sequential_6_layer_call_and_return_conditional_losses_14423564
J__inference_sequential_6_layer_call_and_return_conditional_losses_14423611
J__inference_sequential_6_layer_call_and_return_conditional_losses_14423658?
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
/__inference_sequential_6_layer_call_fn_14423668
/__inference_sequential_6_layer_call_fn_14423678
/__inference_sequential_6_layer_call_fn_14423688
/__inference_sequential_6_layer_call_fn_14423698?
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
J__inference_sequential_7_layer_call_and_return_conditional_losses_14423709
J__inference_sequential_7_layer_call_and_return_conditional_losses_14423720
J__inference_sequential_7_layer_call_and_return_conditional_losses_14423731
J__inference_sequential_7_layer_call_and_return_conditional_losses_14423742?
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
/__inference_sequential_7_layer_call_fn_14423751
/__inference_sequential_7_layer_call_fn_14423760
/__inference_sequential_7_layer_call_fn_14423769
/__inference_sequential_7_layer_call_fn_14423778?
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
&__inference_signature_wrapper_14423328input_1"?
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
I__inference_dense_6_layer_call_and_return_all_conditional_losses_14423795?
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
*__inference_dense_6_layer_call_fn_14423804?
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
__inference_loss_fn_0_14423815?
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
E__inference_dense_7_layer_call_and_return_conditional_losses_14423826?
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
*__inference_dense_7_layer_call_fn_14423835?
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
1__inference_dense_6_activity_regularizer_14422887?
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
E__inference_dense_6_layer_call_and_return_conditional_losses_14423852?
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
#__inference__wrapped_model_14422857o1?.
'?$
"?
input_1??????????
? "4?1
/
output_1#? 
output_1???????????
K__inference_autoencoder_3_layer_call_and_return_conditional_losses_14423285s5?2
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
K__inference_autoencoder_3_layer_call_and_return_conditional_losses_14423307s5?2
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
K__inference_autoencoder_3_layer_call_and_return_conditional_losses_14423382m/?,
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
K__inference_autoencoder_3_layer_call_and_return_conditional_losses_14423436m/?,
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
0__inference_autoencoder_3_layer_call_fn_14423199X5?2
+?(
"?
input_1??????????
p 
? "????????????
0__inference_autoencoder_3_layer_call_fn_14423263X5?2
+?(
"?
input_1??????????
p
? "????????????
0__inference_autoencoder_3_layer_call_fn_14423450R/?,
%?"
?
X??????????
p 
? "????????????
0__inference_autoencoder_3_layer_call_fn_14423464R/?,
%?"
?
X??????????
p
? "???????????d
1__inference_dense_6_activity_regularizer_14422887/$?!
?
?

activation
? "? ?
I__inference_dense_6_layer_call_and_return_all_conditional_losses_14423795l0?-
&?#
!?
inputs??????????
? "4?1
?
0??????????
?
?	
1/0 ?
E__inference_dense_6_layer_call_and_return_conditional_losses_14423852^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_6_layer_call_fn_14423804Q0?-
&?#
!?
inputs??????????
? "????????????
E__inference_dense_7_layer_call_and_return_conditional_losses_14423826^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? 
*__inference_dense_7_layer_call_fn_14423835Q0?-
&?#
!?
inputs??????????
? "???????????=
__inference_loss_fn_0_14423815?

? 
? "? ?
J__inference_sequential_6_layer_call_and_return_conditional_losses_14423517t8?5
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_14423564t8?5
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_14423611{??<
5?2
(?%
dense_6_input??????????
p 

 
? "4?1
?
0??????????
?
?	
1/0 ?
J__inference_sequential_6_layer_call_and_return_conditional_losses_14423658{??<
5?2
(?%
dense_6_input??????????
p

 
? "4?1
?
0??????????
?
?	
1/0 ?
/__inference_sequential_6_layer_call_fn_14423668`??<
5?2
(?%
dense_6_input??????????
p 

 
? "????????????
/__inference_sequential_6_layer_call_fn_14423678Y8?5
.?+
!?
inputs??????????
p 

 
? "????????????
/__inference_sequential_6_layer_call_fn_14423688Y8?5
.?+
!?
inputs??????????
p

 
? "????????????
/__inference_sequential_6_layer_call_fn_14423698`??<
5?2
(?%
dense_6_input??????????
p

 
? "????????????
J__inference_sequential_7_layer_call_and_return_conditional_losses_14423709f8?5
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
J__inference_sequential_7_layer_call_and_return_conditional_losses_14423720f8?5
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
J__inference_sequential_7_layer_call_and_return_conditional_losses_14423731m??<
5?2
(?%
dense_7_input??????????
p 

 
? "&?#
?
0??????????
? ?
J__inference_sequential_7_layer_call_and_return_conditional_losses_14423742m??<
5?2
(?%
dense_7_input??????????
p

 
? "&?#
?
0??????????
? ?
/__inference_sequential_7_layer_call_fn_14423751`??<
5?2
(?%
dense_7_input??????????
p 

 
? "????????????
/__inference_sequential_7_layer_call_fn_14423760Y8?5
.?+
!?
inputs??????????
p 

 
? "????????????
/__inference_sequential_7_layer_call_fn_14423769Y8?5
.?+
!?
inputs??????????
p

 
? "????????????
/__inference_sequential_7_layer_call_fn_14423778`??<
5?2
(?%
dense_7_input??????????
p

 
? "????????????
&__inference_signature_wrapper_14423328z<?9
? 
2?/
-
input_1"?
input_1??????????"4?1
/
output_1#? 
output_1??????????