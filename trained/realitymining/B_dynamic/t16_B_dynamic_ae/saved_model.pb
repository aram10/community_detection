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
dense_122/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:^ *!
shared_namedense_122/kernel
u
$dense_122/kernel/Read/ReadVariableOpReadVariableOpdense_122/kernel*
_output_shapes

:^ *
dtype0
t
dense_122/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_122/bias
m
"dense_122/bias/Read/ReadVariableOpReadVariableOpdense_122/bias*
_output_shapes
: *
dtype0
|
dense_123/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: ^*!
shared_namedense_123/kernel
u
$dense_123/kernel/Read/ReadVariableOpReadVariableOpdense_123/kernel*
_output_shapes

: ^*
dtype0
t
dense_123/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:^*
shared_namedense_123/bias
m
"dense_123/bias/Read/ReadVariableOpReadVariableOpdense_123/bias*
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
VARIABLE_VALUEdense_122/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_122/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_123/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_123/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_122/kerneldense_122/biasdense_123/kerneldense_123/bias*
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
&__inference_signature_wrapper_16652528
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_122/kernel/Read/ReadVariableOp"dense_122/bias/Read/ReadVariableOp$dense_123/kernel/Read/ReadVariableOp"dense_123/bias/Read/ReadVariableOpConst*
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
!__inference__traced_save_16653034
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_122/kerneldense_122/biasdense_123/kerneldense_123/bias*
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
$__inference__traced_restore_16653056??	
?
?
G__inference_dense_123_layer_call_and_return_conditional_losses_16652229

inputs0
matmul_readvariableop_resource: ^-
biasadd_readvariableop_resource:^
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_123/kernel/Regularizer/Square/ReadVariableOp?
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
2dense_123/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_123/kernel/Regularizer/Square/ReadVariableOp?
#dense_123/kernel/Regularizer/SquareSquare:dense_123/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_123/kernel/Regularizer/Square?
"dense_123/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_123/kernel/Regularizer/Const?
 dense_123/kernel/Regularizer/SumSum'dense_123/kernel/Regularizer/Square:y:0+dense_123/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/Sum?
"dense_123/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_123/kernel/Regularizer/mul/x?
 dense_123/kernel/Regularizer/mulMul+dense_123/kernel/Regularizer/mul/x:output:0)dense_123/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_123/kernel/Regularizer/Square/ReadVariableOp*
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
2dense_123/kernel/Regularizer/Square/ReadVariableOp2dense_123/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
L__inference_sequential_123_layer_call_and_return_conditional_losses_16652885
dense_123_input:
(dense_123_matmul_readvariableop_resource: ^7
)dense_123_biasadd_readvariableop_resource:^
identity?? dense_123/BiasAdd/ReadVariableOp?dense_123/MatMul/ReadVariableOp?2dense_123/kernel/Regularizer/Square/ReadVariableOp?
dense_123/MatMul/ReadVariableOpReadVariableOp(dense_123_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02!
dense_123/MatMul/ReadVariableOp?
dense_123/MatMulMatMuldense_123_input'dense_123/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_123/MatMul?
 dense_123/BiasAdd/ReadVariableOpReadVariableOp)dense_123_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02"
 dense_123/BiasAdd/ReadVariableOp?
dense_123/BiasAddBiasAdddense_123/MatMul:product:0(dense_123/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_123/BiasAdd
dense_123/SigmoidSigmoiddense_123/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2
dense_123/Sigmoid?
2dense_123/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_123_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_123/kernel/Regularizer/Square/ReadVariableOp?
#dense_123/kernel/Regularizer/SquareSquare:dense_123/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_123/kernel/Regularizer/Square?
"dense_123/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_123/kernel/Regularizer/Const?
 dense_123/kernel/Regularizer/SumSum'dense_123/kernel/Regularizer/Square:y:0+dense_123/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/Sum?
"dense_123/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_123/kernel/Regularizer/mul/x?
 dense_123/kernel/Regularizer/mulMul+dense_123/kernel/Regularizer/mul/x:output:0)dense_123/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/mul?
IdentityIdentitydense_123/Sigmoid:y:0!^dense_123/BiasAdd/ReadVariableOp ^dense_123/MatMul/ReadVariableOp3^dense_123/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2D
 dense_123/BiasAdd/ReadVariableOp dense_123/BiasAdd/ReadVariableOp2B
dense_123/MatMul/ReadVariableOpdense_123/MatMul/ReadVariableOp2h
2dense_123/kernel/Regularizer/Square/ReadVariableOp2dense_123/kernel/Regularizer/Square/ReadVariableOp:X T
'
_output_shapes
:????????? 
)
_user_specified_namedense_123_input
?
?
G__inference_dense_123_layer_call_and_return_conditional_losses_16652971

inputs0
matmul_readvariableop_resource: ^-
biasadd_readvariableop_resource:^
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_123/kernel/Regularizer/Square/ReadVariableOp?
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
2dense_123/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_123/kernel/Regularizer/Square/ReadVariableOp?
#dense_123/kernel/Regularizer/SquareSquare:dense_123/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_123/kernel/Regularizer/Square?
"dense_123/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_123/kernel/Regularizer/Const?
 dense_123/kernel/Regularizer/SumSum'dense_123/kernel/Regularizer/Square:y:0+dense_123/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/Sum?
"dense_123/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_123/kernel/Regularizer/mul/x?
 dense_123/kernel/Regularizer/mulMul+dense_123/kernel/Regularizer/mul/x:output:0)dense_123/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_123/kernel/Regularizer/Square/ReadVariableOp*
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
2dense_123/kernel/Regularizer/Square/ReadVariableOp2dense_123/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
K__inference_dense_122_layer_call_and_return_all_conditional_losses_16652928

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
G__inference_dense_122_layer_call_and_return_conditional_losses_166520512
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
3__inference_dense_122_activity_regularizer_166520272
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
?
?
__inference_loss_fn_0_16652939M
;dense_122_kernel_regularizer_square_readvariableop_resource:^ 
identity??2dense_122/kernel/Regularizer/Square/ReadVariableOp?
2dense_122/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;dense_122_kernel_regularizer_square_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_122/kernel/Regularizer/Square/ReadVariableOp?
#dense_122/kernel/Regularizer/SquareSquare:dense_122/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_122/kernel/Regularizer/Square?
"dense_122/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_122/kernel/Regularizer/Const?
 dense_122/kernel/Regularizer/SumSum'dense_122/kernel/Regularizer/Square:y:0+dense_122/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/Sum?
"dense_122/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_122/kernel/Regularizer/mul/x?
 dense_122/kernel/Regularizer/mulMul+dense_122/kernel/Regularizer/mul/x:output:0)dense_122/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/mul?
IdentityIdentity$dense_122/kernel/Regularizer/mul:z:03^dense_122/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_122/kernel/Regularizer/Square/ReadVariableOp2dense_122/kernel/Regularizer/Square/ReadVariableOp
?B
?
L__inference_sequential_122_layer_call_and_return_conditional_losses_16652792

inputs:
(dense_122_matmul_readvariableop_resource:^ 7
)dense_122_biasadd_readvariableop_resource: 
identity

identity_1?? dense_122/BiasAdd/ReadVariableOp?dense_122/MatMul/ReadVariableOp?2dense_122/kernel/Regularizer/Square/ReadVariableOp?
dense_122/MatMul/ReadVariableOpReadVariableOp(dense_122_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02!
dense_122/MatMul/ReadVariableOp?
dense_122/MatMulMatMulinputs'dense_122/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_122/MatMul?
 dense_122/BiasAdd/ReadVariableOpReadVariableOp)dense_122_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_122/BiasAdd/ReadVariableOp?
dense_122/BiasAddBiasAdddense_122/MatMul:product:0(dense_122/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_122/BiasAdd
dense_122/SigmoidSigmoiddense_122/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_122/Sigmoid?
4dense_122/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 26
4dense_122/ActivityRegularizer/Mean/reduction_indices?
"dense_122/ActivityRegularizer/MeanMeandense_122/Sigmoid:y:0=dense_122/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2$
"dense_122/ActivityRegularizer/Mean?
'dense_122/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2)
'dense_122/ActivityRegularizer/Maximum/y?
%dense_122/ActivityRegularizer/MaximumMaximum+dense_122/ActivityRegularizer/Mean:output:00dense_122/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2'
%dense_122/ActivityRegularizer/Maximum?
'dense_122/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2)
'dense_122/ActivityRegularizer/truediv/x?
%dense_122/ActivityRegularizer/truedivRealDiv0dense_122/ActivityRegularizer/truediv/x:output:0)dense_122/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2'
%dense_122/ActivityRegularizer/truediv?
!dense_122/ActivityRegularizer/LogLog)dense_122/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2#
!dense_122/ActivityRegularizer/Log?
#dense_122/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_122/ActivityRegularizer/mul/x?
!dense_122/ActivityRegularizer/mulMul,dense_122/ActivityRegularizer/mul/x:output:0%dense_122/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2#
!dense_122/ActivityRegularizer/mul?
#dense_122/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#dense_122/ActivityRegularizer/sub/x?
!dense_122/ActivityRegularizer/subSub,dense_122/ActivityRegularizer/sub/x:output:0)dense_122/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2#
!dense_122/ActivityRegularizer/sub?
)dense_122/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2+
)dense_122/ActivityRegularizer/truediv_1/x?
'dense_122/ActivityRegularizer/truediv_1RealDiv2dense_122/ActivityRegularizer/truediv_1/x:output:0%dense_122/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2)
'dense_122/ActivityRegularizer/truediv_1?
#dense_122/ActivityRegularizer/Log_1Log+dense_122/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2%
#dense_122/ActivityRegularizer/Log_1?
%dense_122/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2'
%dense_122/ActivityRegularizer/mul_1/x?
#dense_122/ActivityRegularizer/mul_1Mul.dense_122/ActivityRegularizer/mul_1/x:output:0'dense_122/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2%
#dense_122/ActivityRegularizer/mul_1?
!dense_122/ActivityRegularizer/addAddV2%dense_122/ActivityRegularizer/mul:z:0'dense_122/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_122/ActivityRegularizer/add?
#dense_122/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#dense_122/ActivityRegularizer/Const?
!dense_122/ActivityRegularizer/SumSum%dense_122/ActivityRegularizer/add:z:0,dense_122/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_122/ActivityRegularizer/Sum?
%dense_122/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%dense_122/ActivityRegularizer/mul_2/x?
#dense_122/ActivityRegularizer/mul_2Mul.dense_122/ActivityRegularizer/mul_2/x:output:0*dense_122/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_122/ActivityRegularizer/mul_2?
#dense_122/ActivityRegularizer/ShapeShapedense_122/Sigmoid:y:0*
T0*
_output_shapes
:2%
#dense_122/ActivityRegularizer/Shape?
1dense_122/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_122/ActivityRegularizer/strided_slice/stack?
3dense_122/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_122/ActivityRegularizer/strided_slice/stack_1?
3dense_122/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_122/ActivityRegularizer/strided_slice/stack_2?
+dense_122/ActivityRegularizer/strided_sliceStridedSlice,dense_122/ActivityRegularizer/Shape:output:0:dense_122/ActivityRegularizer/strided_slice/stack:output:0<dense_122/ActivityRegularizer/strided_slice/stack_1:output:0<dense_122/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_122/ActivityRegularizer/strided_slice?
"dense_122/ActivityRegularizer/CastCast4dense_122/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_122/ActivityRegularizer/Cast?
'dense_122/ActivityRegularizer/truediv_2RealDiv'dense_122/ActivityRegularizer/mul_2:z:0&dense_122/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2)
'dense_122/ActivityRegularizer/truediv_2?
2dense_122/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_122_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_122/kernel/Regularizer/Square/ReadVariableOp?
#dense_122/kernel/Regularizer/SquareSquare:dense_122/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_122/kernel/Regularizer/Square?
"dense_122/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_122/kernel/Regularizer/Const?
 dense_122/kernel/Regularizer/SumSum'dense_122/kernel/Regularizer/Square:y:0+dense_122/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/Sum?
"dense_122/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_122/kernel/Regularizer/mul/x?
 dense_122/kernel/Regularizer/mulMul+dense_122/kernel/Regularizer/mul/x:output:0)dense_122/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/mul?
IdentityIdentitydense_122/Sigmoid:y:0!^dense_122/BiasAdd/ReadVariableOp ^dense_122/MatMul/ReadVariableOp3^dense_122/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity+dense_122/ActivityRegularizer/truediv_2:z:0!^dense_122/BiasAdd/ReadVariableOp ^dense_122/MatMul/ReadVariableOp3^dense_122/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2D
 dense_122/BiasAdd/ReadVariableOp dense_122/BiasAdd/ReadVariableOp2B
dense_122/MatMul/ReadVariableOpdense_122/MatMul/ReadVariableOp2h
2dense_122/kernel/Regularizer/Square/ReadVariableOp2dense_122/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
,__inference_dense_123_layer_call_fn_16652954

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
G__inference_dense_123_layer_call_and_return_conditional_losses_166522292
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
L__inference_sequential_123_layer_call_and_return_conditional_losses_16652902
dense_123_input:
(dense_123_matmul_readvariableop_resource: ^7
)dense_123_biasadd_readvariableop_resource:^
identity?? dense_123/BiasAdd/ReadVariableOp?dense_123/MatMul/ReadVariableOp?2dense_123/kernel/Regularizer/Square/ReadVariableOp?
dense_123/MatMul/ReadVariableOpReadVariableOp(dense_123_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02!
dense_123/MatMul/ReadVariableOp?
dense_123/MatMulMatMuldense_123_input'dense_123/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_123/MatMul?
 dense_123/BiasAdd/ReadVariableOpReadVariableOp)dense_123_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02"
 dense_123/BiasAdd/ReadVariableOp?
dense_123/BiasAddBiasAdddense_123/MatMul:product:0(dense_123/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_123/BiasAdd
dense_123/SigmoidSigmoiddense_123/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2
dense_123/Sigmoid?
2dense_123/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_123_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_123/kernel/Regularizer/Square/ReadVariableOp?
#dense_123/kernel/Regularizer/SquareSquare:dense_123/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_123/kernel/Regularizer/Square?
"dense_123/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_123/kernel/Regularizer/Const?
 dense_123/kernel/Regularizer/SumSum'dense_123/kernel/Regularizer/Square:y:0+dense_123/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/Sum?
"dense_123/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_123/kernel/Regularizer/mul/x?
 dense_123/kernel/Regularizer/mulMul+dense_123/kernel/Regularizer/mul/x:output:0)dense_123/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/mul?
IdentityIdentitydense_123/Sigmoid:y:0!^dense_123/BiasAdd/ReadVariableOp ^dense_123/MatMul/ReadVariableOp3^dense_123/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2D
 dense_123/BiasAdd/ReadVariableOp dense_123/BiasAdd/ReadVariableOp2B
dense_123/MatMul/ReadVariableOpdense_123/MatMul/ReadVariableOp2h
2dense_123/kernel/Regularizer/Square/ReadVariableOp2dense_123/kernel/Regularizer/Square/ReadVariableOp:X T
'
_output_shapes
:????????? 
)
_user_specified_namedense_123_input
?
?
1__inference_autoencoder_61_layer_call_fn_16652445
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
L__inference_autoencoder_61_layer_call_and_return_conditional_losses_166524192
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
?#
?
L__inference_sequential_122_layer_call_and_return_conditional_losses_16652205
input_62$
dense_122_16652184:^  
dense_122_16652186: 
identity

identity_1??!dense_122/StatefulPartitionedCall?2dense_122/kernel/Regularizer/Square/ReadVariableOp?
!dense_122/StatefulPartitionedCallStatefulPartitionedCallinput_62dense_122_16652184dense_122_16652186*
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
G__inference_dense_122_layer_call_and_return_conditional_losses_166520512#
!dense_122/StatefulPartitionedCall?
-dense_122/ActivityRegularizer/PartitionedCallPartitionedCall*dense_122/StatefulPartitionedCall:output:0*
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
3__inference_dense_122_activity_regularizer_166520272/
-dense_122/ActivityRegularizer/PartitionedCall?
#dense_122/ActivityRegularizer/ShapeShape*dense_122/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2%
#dense_122/ActivityRegularizer/Shape?
1dense_122/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_122/ActivityRegularizer/strided_slice/stack?
3dense_122/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_122/ActivityRegularizer/strided_slice/stack_1?
3dense_122/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_122/ActivityRegularizer/strided_slice/stack_2?
+dense_122/ActivityRegularizer/strided_sliceStridedSlice,dense_122/ActivityRegularizer/Shape:output:0:dense_122/ActivityRegularizer/strided_slice/stack:output:0<dense_122/ActivityRegularizer/strided_slice/stack_1:output:0<dense_122/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_122/ActivityRegularizer/strided_slice?
"dense_122/ActivityRegularizer/CastCast4dense_122/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_122/ActivityRegularizer/Cast?
%dense_122/ActivityRegularizer/truedivRealDiv6dense_122/ActivityRegularizer/PartitionedCall:output:0&dense_122/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_122/ActivityRegularizer/truediv?
2dense_122/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_122_16652184*
_output_shapes

:^ *
dtype024
2dense_122/kernel/Regularizer/Square/ReadVariableOp?
#dense_122/kernel/Regularizer/SquareSquare:dense_122/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_122/kernel/Regularizer/Square?
"dense_122/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_122/kernel/Regularizer/Const?
 dense_122/kernel/Regularizer/SumSum'dense_122/kernel/Regularizer/Square:y:0+dense_122/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/Sum?
"dense_122/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_122/kernel/Regularizer/mul/x?
 dense_122/kernel/Regularizer/mulMul+dense_122/kernel/Regularizer/mul/x:output:0)dense_122/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/mul?
IdentityIdentity*dense_122/StatefulPartitionedCall:output:0"^dense_122/StatefulPartitionedCall3^dense_122/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity)dense_122/ActivityRegularizer/truediv:z:0"^dense_122/StatefulPartitionedCall3^dense_122/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2F
!dense_122/StatefulPartitionedCall!dense_122/StatefulPartitionedCall2h
2dense_122/kernel/Regularizer/Square/ReadVariableOp2dense_122/kernel/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:?????????^
"
_user_specified_name
input_62
?
S
3__inference_dense_122_activity_regularizer_16652027

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
1__inference_autoencoder_61_layer_call_fn_16652556
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
L__inference_autoencoder_61_layer_call_and_return_conditional_losses_166524192
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
?
?
L__inference_sequential_123_layer_call_and_return_conditional_losses_16652242

inputs$
dense_123_16652230: ^ 
dense_123_16652232:^
identity??!dense_123/StatefulPartitionedCall?2dense_123/kernel/Regularizer/Square/ReadVariableOp?
!dense_123/StatefulPartitionedCallStatefulPartitionedCallinputsdense_123_16652230dense_123_16652232*
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
G__inference_dense_123_layer_call_and_return_conditional_losses_166522292#
!dense_123/StatefulPartitionedCall?
2dense_123/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_123_16652230*
_output_shapes

: ^*
dtype024
2dense_123/kernel/Regularizer/Square/ReadVariableOp?
#dense_123/kernel/Regularizer/SquareSquare:dense_123/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_123/kernel/Regularizer/Square?
"dense_123/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_123/kernel/Regularizer/Const?
 dense_123/kernel/Regularizer/SumSum'dense_123/kernel/Regularizer/Square:y:0+dense_123/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/Sum?
"dense_123/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_123/kernel/Regularizer/mul/x?
 dense_123/kernel/Regularizer/mulMul+dense_123/kernel/Regularizer/mul/x:output:0)dense_123/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/mul?
IdentityIdentity*dense_123/StatefulPartitionedCall:output:0"^dense_123/StatefulPartitionedCall3^dense_123/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2F
!dense_123/StatefulPartitionedCall!dense_123/StatefulPartitionedCall2h
2dense_123/kernel/Regularizer/Square/ReadVariableOp2dense_123/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
1__inference_sequential_123_layer_call_fn_16652816

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
L__inference_sequential_123_layer_call_and_return_conditional_losses_166522422
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
1__inference_sequential_122_layer_call_fn_16652690

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
L__inference_sequential_122_layer_call_and_return_conditional_losses_166520732
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
?h
?
L__inference_autoencoder_61_layer_call_and_return_conditional_losses_16652674
xI
7sequential_122_dense_122_matmul_readvariableop_resource:^ F
8sequential_122_dense_122_biasadd_readvariableop_resource: I
7sequential_123_dense_123_matmul_readvariableop_resource: ^F
8sequential_123_dense_123_biasadd_readvariableop_resource:^
identity

identity_1??2dense_122/kernel/Regularizer/Square/ReadVariableOp?2dense_123/kernel/Regularizer/Square/ReadVariableOp?/sequential_122/dense_122/BiasAdd/ReadVariableOp?.sequential_122/dense_122/MatMul/ReadVariableOp?/sequential_123/dense_123/BiasAdd/ReadVariableOp?.sequential_123/dense_123/MatMul/ReadVariableOp?
.sequential_122/dense_122/MatMul/ReadVariableOpReadVariableOp7sequential_122_dense_122_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype020
.sequential_122/dense_122/MatMul/ReadVariableOp?
sequential_122/dense_122/MatMulMatMulx6sequential_122/dense_122/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2!
sequential_122/dense_122/MatMul?
/sequential_122/dense_122/BiasAdd/ReadVariableOpReadVariableOp8sequential_122_dense_122_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential_122/dense_122/BiasAdd/ReadVariableOp?
 sequential_122/dense_122/BiasAddBiasAdd)sequential_122/dense_122/MatMul:product:07sequential_122/dense_122/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2"
 sequential_122/dense_122/BiasAdd?
 sequential_122/dense_122/SigmoidSigmoid)sequential_122/dense_122/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2"
 sequential_122/dense_122/Sigmoid?
Csequential_122/dense_122/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2E
Csequential_122/dense_122/ActivityRegularizer/Mean/reduction_indices?
1sequential_122/dense_122/ActivityRegularizer/MeanMean$sequential_122/dense_122/Sigmoid:y:0Lsequential_122/dense_122/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 23
1sequential_122/dense_122/ActivityRegularizer/Mean?
6sequential_122/dense_122/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.28
6sequential_122/dense_122/ActivityRegularizer/Maximum/y?
4sequential_122/dense_122/ActivityRegularizer/MaximumMaximum:sequential_122/dense_122/ActivityRegularizer/Mean:output:0?sequential_122/dense_122/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 26
4sequential_122/dense_122/ActivityRegularizer/Maximum?
6sequential_122/dense_122/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<28
6sequential_122/dense_122/ActivityRegularizer/truediv/x?
4sequential_122/dense_122/ActivityRegularizer/truedivRealDiv?sequential_122/dense_122/ActivityRegularizer/truediv/x:output:08sequential_122/dense_122/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 26
4sequential_122/dense_122/ActivityRegularizer/truediv?
0sequential_122/dense_122/ActivityRegularizer/LogLog8sequential_122/dense_122/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 22
0sequential_122/dense_122/ActivityRegularizer/Log?
2sequential_122/dense_122/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<24
2sequential_122/dense_122/ActivityRegularizer/mul/x?
0sequential_122/dense_122/ActivityRegularizer/mulMul;sequential_122/dense_122/ActivityRegularizer/mul/x:output:04sequential_122/dense_122/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 22
0sequential_122/dense_122/ActivityRegularizer/mul?
2sequential_122/dense_122/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2sequential_122/dense_122/ActivityRegularizer/sub/x?
0sequential_122/dense_122/ActivityRegularizer/subSub;sequential_122/dense_122/ActivityRegularizer/sub/x:output:08sequential_122/dense_122/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 22
0sequential_122/dense_122/ActivityRegularizer/sub?
8sequential_122/dense_122/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2:
8sequential_122/dense_122/ActivityRegularizer/truediv_1/x?
6sequential_122/dense_122/ActivityRegularizer/truediv_1RealDivAsequential_122/dense_122/ActivityRegularizer/truediv_1/x:output:04sequential_122/dense_122/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 28
6sequential_122/dense_122/ActivityRegularizer/truediv_1?
2sequential_122/dense_122/ActivityRegularizer/Log_1Log:sequential_122/dense_122/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 24
2sequential_122/dense_122/ActivityRegularizer/Log_1?
4sequential_122/dense_122/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?26
4sequential_122/dense_122/ActivityRegularizer/mul_1/x?
2sequential_122/dense_122/ActivityRegularizer/mul_1Mul=sequential_122/dense_122/ActivityRegularizer/mul_1/x:output:06sequential_122/dense_122/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 24
2sequential_122/dense_122/ActivityRegularizer/mul_1?
0sequential_122/dense_122/ActivityRegularizer/addAddV24sequential_122/dense_122/ActivityRegularizer/mul:z:06sequential_122/dense_122/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 22
0sequential_122/dense_122/ActivityRegularizer/add?
2sequential_122/dense_122/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 24
2sequential_122/dense_122/ActivityRegularizer/Const?
0sequential_122/dense_122/ActivityRegularizer/SumSum4sequential_122/dense_122/ActivityRegularizer/add:z:0;sequential_122/dense_122/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 22
0sequential_122/dense_122/ActivityRegularizer/Sum?
4sequential_122/dense_122/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??26
4sequential_122/dense_122/ActivityRegularizer/mul_2/x?
2sequential_122/dense_122/ActivityRegularizer/mul_2Mul=sequential_122/dense_122/ActivityRegularizer/mul_2/x:output:09sequential_122/dense_122/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 24
2sequential_122/dense_122/ActivityRegularizer/mul_2?
2sequential_122/dense_122/ActivityRegularizer/ShapeShape$sequential_122/dense_122/Sigmoid:y:0*
T0*
_output_shapes
:24
2sequential_122/dense_122/ActivityRegularizer/Shape?
@sequential_122/dense_122/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential_122/dense_122/ActivityRegularizer/strided_slice/stack?
Bsequential_122/dense_122/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_122/dense_122/ActivityRegularizer/strided_slice/stack_1?
Bsequential_122/dense_122/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_122/dense_122/ActivityRegularizer/strided_slice/stack_2?
:sequential_122/dense_122/ActivityRegularizer/strided_sliceStridedSlice;sequential_122/dense_122/ActivityRegularizer/Shape:output:0Isequential_122/dense_122/ActivityRegularizer/strided_slice/stack:output:0Ksequential_122/dense_122/ActivityRegularizer/strided_slice/stack_1:output:0Ksequential_122/dense_122/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_122/dense_122/ActivityRegularizer/strided_slice?
1sequential_122/dense_122/ActivityRegularizer/CastCastCsequential_122/dense_122/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 23
1sequential_122/dense_122/ActivityRegularizer/Cast?
6sequential_122/dense_122/ActivityRegularizer/truediv_2RealDiv6sequential_122/dense_122/ActivityRegularizer/mul_2:z:05sequential_122/dense_122/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 28
6sequential_122/dense_122/ActivityRegularizer/truediv_2?
.sequential_123/dense_123/MatMul/ReadVariableOpReadVariableOp7sequential_123_dense_123_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype020
.sequential_123/dense_123/MatMul/ReadVariableOp?
sequential_123/dense_123/MatMulMatMul$sequential_122/dense_122/Sigmoid:y:06sequential_123/dense_123/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2!
sequential_123/dense_123/MatMul?
/sequential_123/dense_123/BiasAdd/ReadVariableOpReadVariableOp8sequential_123_dense_123_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype021
/sequential_123/dense_123/BiasAdd/ReadVariableOp?
 sequential_123/dense_123/BiasAddBiasAdd)sequential_123/dense_123/MatMul:product:07sequential_123/dense_123/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2"
 sequential_123/dense_123/BiasAdd?
 sequential_123/dense_123/SigmoidSigmoid)sequential_123/dense_123/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2"
 sequential_123/dense_123/Sigmoid?
2dense_122/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7sequential_122_dense_122_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_122/kernel/Regularizer/Square/ReadVariableOp?
#dense_122/kernel/Regularizer/SquareSquare:dense_122/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_122/kernel/Regularizer/Square?
"dense_122/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_122/kernel/Regularizer/Const?
 dense_122/kernel/Regularizer/SumSum'dense_122/kernel/Regularizer/Square:y:0+dense_122/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/Sum?
"dense_122/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_122/kernel/Regularizer/mul/x?
 dense_122/kernel/Regularizer/mulMul+dense_122/kernel/Regularizer/mul/x:output:0)dense_122/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/mul?
2dense_123/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7sequential_123_dense_123_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_123/kernel/Regularizer/Square/ReadVariableOp?
#dense_123/kernel/Regularizer/SquareSquare:dense_123/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_123/kernel/Regularizer/Square?
"dense_123/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_123/kernel/Regularizer/Const?
 dense_123/kernel/Regularizer/SumSum'dense_123/kernel/Regularizer/Square:y:0+dense_123/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/Sum?
"dense_123/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_123/kernel/Regularizer/mul/x?
 dense_123/kernel/Regularizer/mulMul+dense_123/kernel/Regularizer/mul/x:output:0)dense_123/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/mul?
IdentityIdentity$sequential_123/dense_123/Sigmoid:y:03^dense_122/kernel/Regularizer/Square/ReadVariableOp3^dense_123/kernel/Regularizer/Square/ReadVariableOp0^sequential_122/dense_122/BiasAdd/ReadVariableOp/^sequential_122/dense_122/MatMul/ReadVariableOp0^sequential_123/dense_123/BiasAdd/ReadVariableOp/^sequential_123/dense_123/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity:sequential_122/dense_122/ActivityRegularizer/truediv_2:z:03^dense_122/kernel/Regularizer/Square/ReadVariableOp3^dense_123/kernel/Regularizer/Square/ReadVariableOp0^sequential_122/dense_122/BiasAdd/ReadVariableOp/^sequential_122/dense_122/MatMul/ReadVariableOp0^sequential_123/dense_123/BiasAdd/ReadVariableOp/^sequential_123/dense_123/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_122/kernel/Regularizer/Square/ReadVariableOp2dense_122/kernel/Regularizer/Square/ReadVariableOp2h
2dense_123/kernel/Regularizer/Square/ReadVariableOp2dense_123/kernel/Regularizer/Square/ReadVariableOp2b
/sequential_122/dense_122/BiasAdd/ReadVariableOp/sequential_122/dense_122/BiasAdd/ReadVariableOp2`
.sequential_122/dense_122/MatMul/ReadVariableOp.sequential_122/dense_122/MatMul/ReadVariableOp2b
/sequential_123/dense_123/BiasAdd/ReadVariableOp/sequential_123/dense_123/BiasAdd/ReadVariableOp2`
.sequential_123/dense_123/MatMul/ReadVariableOp.sequential_123/dense_123/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?
?
G__inference_dense_122_layer_call_and_return_conditional_losses_16652999

inputs0
matmul_readvariableop_resource:^ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_122/kernel/Regularizer/Square/ReadVariableOp?
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
2dense_122/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_122/kernel/Regularizer/Square/ReadVariableOp?
#dense_122/kernel/Regularizer/SquareSquare:dense_122/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_122/kernel/Regularizer/Square?
"dense_122/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_122/kernel/Regularizer/Const?
 dense_122/kernel/Regularizer/SumSum'dense_122/kernel/Regularizer/Square:y:0+dense_122/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/Sum?
"dense_122/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_122/kernel/Regularizer/mul/x?
 dense_122/kernel/Regularizer/mulMul+dense_122/kernel/Regularizer/mul/x:output:0)dense_122/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_122/kernel/Regularizer/Square/ReadVariableOp*
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
2dense_122/kernel/Regularizer/Square/ReadVariableOp2dense_122/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
!__inference__traced_save_16653034
file_prefix/
+savev2_dense_122_kernel_read_readvariableop-
)savev2_dense_122_bias_read_readvariableop/
+savev2_dense_123_kernel_read_readvariableop-
)savev2_dense_123_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_122_kernel_read_readvariableop)savev2_dense_122_bias_read_readvariableop+savev2_dense_123_kernel_read_readvariableop)savev2_dense_123_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
?#
?
L__inference_sequential_122_layer_call_and_return_conditional_losses_16652139

inputs$
dense_122_16652118:^  
dense_122_16652120: 
identity

identity_1??!dense_122/StatefulPartitionedCall?2dense_122/kernel/Regularizer/Square/ReadVariableOp?
!dense_122/StatefulPartitionedCallStatefulPartitionedCallinputsdense_122_16652118dense_122_16652120*
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
G__inference_dense_122_layer_call_and_return_conditional_losses_166520512#
!dense_122/StatefulPartitionedCall?
-dense_122/ActivityRegularizer/PartitionedCallPartitionedCall*dense_122/StatefulPartitionedCall:output:0*
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
3__inference_dense_122_activity_regularizer_166520272/
-dense_122/ActivityRegularizer/PartitionedCall?
#dense_122/ActivityRegularizer/ShapeShape*dense_122/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2%
#dense_122/ActivityRegularizer/Shape?
1dense_122/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_122/ActivityRegularizer/strided_slice/stack?
3dense_122/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_122/ActivityRegularizer/strided_slice/stack_1?
3dense_122/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_122/ActivityRegularizer/strided_slice/stack_2?
+dense_122/ActivityRegularizer/strided_sliceStridedSlice,dense_122/ActivityRegularizer/Shape:output:0:dense_122/ActivityRegularizer/strided_slice/stack:output:0<dense_122/ActivityRegularizer/strided_slice/stack_1:output:0<dense_122/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_122/ActivityRegularizer/strided_slice?
"dense_122/ActivityRegularizer/CastCast4dense_122/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_122/ActivityRegularizer/Cast?
%dense_122/ActivityRegularizer/truedivRealDiv6dense_122/ActivityRegularizer/PartitionedCall:output:0&dense_122/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_122/ActivityRegularizer/truediv?
2dense_122/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_122_16652118*
_output_shapes

:^ *
dtype024
2dense_122/kernel/Regularizer/Square/ReadVariableOp?
#dense_122/kernel/Regularizer/SquareSquare:dense_122/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_122/kernel/Regularizer/Square?
"dense_122/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_122/kernel/Regularizer/Const?
 dense_122/kernel/Regularizer/SumSum'dense_122/kernel/Regularizer/Square:y:0+dense_122/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/Sum?
"dense_122/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_122/kernel/Regularizer/mul/x?
 dense_122/kernel/Regularizer/mulMul+dense_122/kernel/Regularizer/mul/x:output:0)dense_122/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/mul?
IdentityIdentity*dense_122/StatefulPartitionedCall:output:0"^dense_122/StatefulPartitionedCall3^dense_122/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity)dense_122/ActivityRegularizer/truediv:z:0"^dense_122/StatefulPartitionedCall3^dense_122/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2F
!dense_122/StatefulPartitionedCall!dense_122/StatefulPartitionedCall2h
2dense_122/kernel/Regularizer/Square/ReadVariableOp2dense_122/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
1__inference_autoencoder_61_layer_call_fn_16652542
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
L__inference_autoencoder_61_layer_call_and_return_conditional_losses_166523632
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
L__inference_autoencoder_61_layer_call_and_return_conditional_losses_16652419
x)
sequential_122_16652394:^ %
sequential_122_16652396: )
sequential_123_16652400: ^%
sequential_123_16652402:^
identity

identity_1??2dense_122/kernel/Regularizer/Square/ReadVariableOp?2dense_123/kernel/Regularizer/Square/ReadVariableOp?&sequential_122/StatefulPartitionedCall?&sequential_123/StatefulPartitionedCall?
&sequential_122/StatefulPartitionedCallStatefulPartitionedCallxsequential_122_16652394sequential_122_16652396*
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
L__inference_sequential_122_layer_call_and_return_conditional_losses_166521392(
&sequential_122/StatefulPartitionedCall?
&sequential_123/StatefulPartitionedCallStatefulPartitionedCall/sequential_122/StatefulPartitionedCall:output:0sequential_123_16652400sequential_123_16652402*
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
L__inference_sequential_123_layer_call_and_return_conditional_losses_166522852(
&sequential_123/StatefulPartitionedCall?
2dense_122/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_122_16652394*
_output_shapes

:^ *
dtype024
2dense_122/kernel/Regularizer/Square/ReadVariableOp?
#dense_122/kernel/Regularizer/SquareSquare:dense_122/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_122/kernel/Regularizer/Square?
"dense_122/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_122/kernel/Regularizer/Const?
 dense_122/kernel/Regularizer/SumSum'dense_122/kernel/Regularizer/Square:y:0+dense_122/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/Sum?
"dense_122/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_122/kernel/Regularizer/mul/x?
 dense_122/kernel/Regularizer/mulMul+dense_122/kernel/Regularizer/mul/x:output:0)dense_122/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/mul?
2dense_123/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_123_16652400*
_output_shapes

: ^*
dtype024
2dense_123/kernel/Regularizer/Square/ReadVariableOp?
#dense_123/kernel/Regularizer/SquareSquare:dense_123/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_123/kernel/Regularizer/Square?
"dense_123/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_123/kernel/Regularizer/Const?
 dense_123/kernel/Regularizer/SumSum'dense_123/kernel/Regularizer/Square:y:0+dense_123/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/Sum?
"dense_123/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_123/kernel/Regularizer/mul/x?
 dense_123/kernel/Regularizer/mulMul+dense_123/kernel/Regularizer/mul/x:output:0)dense_123/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/mul?
IdentityIdentity/sequential_123/StatefulPartitionedCall:output:03^dense_122/kernel/Regularizer/Square/ReadVariableOp3^dense_123/kernel/Regularizer/Square/ReadVariableOp'^sequential_122/StatefulPartitionedCall'^sequential_123/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity/sequential_122/StatefulPartitionedCall:output:13^dense_122/kernel/Regularizer/Square/ReadVariableOp3^dense_123/kernel/Regularizer/Square/ReadVariableOp'^sequential_122/StatefulPartitionedCall'^sequential_123/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_122/kernel/Regularizer/Square/ReadVariableOp2dense_122/kernel/Regularizer/Square/ReadVariableOp2h
2dense_123/kernel/Regularizer/Square/ReadVariableOp2dense_123/kernel/Regularizer/Square/ReadVariableOp2P
&sequential_122/StatefulPartitionedCall&sequential_122/StatefulPartitionedCall2P
&sequential_123/StatefulPartitionedCall&sequential_123/StatefulPartitionedCall:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?#
?
L__inference_sequential_122_layer_call_and_return_conditional_losses_16652181
input_62$
dense_122_16652160:^  
dense_122_16652162: 
identity

identity_1??!dense_122/StatefulPartitionedCall?2dense_122/kernel/Regularizer/Square/ReadVariableOp?
!dense_122/StatefulPartitionedCallStatefulPartitionedCallinput_62dense_122_16652160dense_122_16652162*
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
G__inference_dense_122_layer_call_and_return_conditional_losses_166520512#
!dense_122/StatefulPartitionedCall?
-dense_122/ActivityRegularizer/PartitionedCallPartitionedCall*dense_122/StatefulPartitionedCall:output:0*
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
3__inference_dense_122_activity_regularizer_166520272/
-dense_122/ActivityRegularizer/PartitionedCall?
#dense_122/ActivityRegularizer/ShapeShape*dense_122/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2%
#dense_122/ActivityRegularizer/Shape?
1dense_122/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_122/ActivityRegularizer/strided_slice/stack?
3dense_122/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_122/ActivityRegularizer/strided_slice/stack_1?
3dense_122/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_122/ActivityRegularizer/strided_slice/stack_2?
+dense_122/ActivityRegularizer/strided_sliceStridedSlice,dense_122/ActivityRegularizer/Shape:output:0:dense_122/ActivityRegularizer/strided_slice/stack:output:0<dense_122/ActivityRegularizer/strided_slice/stack_1:output:0<dense_122/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_122/ActivityRegularizer/strided_slice?
"dense_122/ActivityRegularizer/CastCast4dense_122/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_122/ActivityRegularizer/Cast?
%dense_122/ActivityRegularizer/truedivRealDiv6dense_122/ActivityRegularizer/PartitionedCall:output:0&dense_122/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_122/ActivityRegularizer/truediv?
2dense_122/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_122_16652160*
_output_shapes

:^ *
dtype024
2dense_122/kernel/Regularizer/Square/ReadVariableOp?
#dense_122/kernel/Regularizer/SquareSquare:dense_122/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_122/kernel/Regularizer/Square?
"dense_122/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_122/kernel/Regularizer/Const?
 dense_122/kernel/Regularizer/SumSum'dense_122/kernel/Regularizer/Square:y:0+dense_122/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/Sum?
"dense_122/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_122/kernel/Regularizer/mul/x?
 dense_122/kernel/Regularizer/mulMul+dense_122/kernel/Regularizer/mul/x:output:0)dense_122/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/mul?
IdentityIdentity*dense_122/StatefulPartitionedCall:output:0"^dense_122/StatefulPartitionedCall3^dense_122/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity)dense_122/ActivityRegularizer/truediv:z:0"^dense_122/StatefulPartitionedCall3^dense_122/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2F
!dense_122/StatefulPartitionedCall!dense_122/StatefulPartitionedCall2h
2dense_122/kernel/Regularizer/Square/ReadVariableOp2dense_122/kernel/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:?????????^
"
_user_specified_name
input_62
?
?
1__inference_sequential_123_layer_call_fn_16652807
dense_123_input
unknown: ^
	unknown_0:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_123_inputunknown	unknown_0*
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
L__inference_sequential_123_layer_call_and_return_conditional_losses_166522422
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
_user_specified_namedense_123_input
?
?
,__inference_dense_122_layer_call_fn_16652917

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
G__inference_dense_122_layer_call_and_return_conditional_losses_166520512
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
?
?
1__inference_sequential_122_layer_call_fn_16652081
input_62
unknown:^ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_62unknown	unknown_0*
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
L__inference_sequential_122_layer_call_and_return_conditional_losses_166520732
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
input_62
?h
?
L__inference_autoencoder_61_layer_call_and_return_conditional_losses_16652615
xI
7sequential_122_dense_122_matmul_readvariableop_resource:^ F
8sequential_122_dense_122_biasadd_readvariableop_resource: I
7sequential_123_dense_123_matmul_readvariableop_resource: ^F
8sequential_123_dense_123_biasadd_readvariableop_resource:^
identity

identity_1??2dense_122/kernel/Regularizer/Square/ReadVariableOp?2dense_123/kernel/Regularizer/Square/ReadVariableOp?/sequential_122/dense_122/BiasAdd/ReadVariableOp?.sequential_122/dense_122/MatMul/ReadVariableOp?/sequential_123/dense_123/BiasAdd/ReadVariableOp?.sequential_123/dense_123/MatMul/ReadVariableOp?
.sequential_122/dense_122/MatMul/ReadVariableOpReadVariableOp7sequential_122_dense_122_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype020
.sequential_122/dense_122/MatMul/ReadVariableOp?
sequential_122/dense_122/MatMulMatMulx6sequential_122/dense_122/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2!
sequential_122/dense_122/MatMul?
/sequential_122/dense_122/BiasAdd/ReadVariableOpReadVariableOp8sequential_122_dense_122_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential_122/dense_122/BiasAdd/ReadVariableOp?
 sequential_122/dense_122/BiasAddBiasAdd)sequential_122/dense_122/MatMul:product:07sequential_122/dense_122/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2"
 sequential_122/dense_122/BiasAdd?
 sequential_122/dense_122/SigmoidSigmoid)sequential_122/dense_122/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2"
 sequential_122/dense_122/Sigmoid?
Csequential_122/dense_122/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2E
Csequential_122/dense_122/ActivityRegularizer/Mean/reduction_indices?
1sequential_122/dense_122/ActivityRegularizer/MeanMean$sequential_122/dense_122/Sigmoid:y:0Lsequential_122/dense_122/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 23
1sequential_122/dense_122/ActivityRegularizer/Mean?
6sequential_122/dense_122/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.28
6sequential_122/dense_122/ActivityRegularizer/Maximum/y?
4sequential_122/dense_122/ActivityRegularizer/MaximumMaximum:sequential_122/dense_122/ActivityRegularizer/Mean:output:0?sequential_122/dense_122/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 26
4sequential_122/dense_122/ActivityRegularizer/Maximum?
6sequential_122/dense_122/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<28
6sequential_122/dense_122/ActivityRegularizer/truediv/x?
4sequential_122/dense_122/ActivityRegularizer/truedivRealDiv?sequential_122/dense_122/ActivityRegularizer/truediv/x:output:08sequential_122/dense_122/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 26
4sequential_122/dense_122/ActivityRegularizer/truediv?
0sequential_122/dense_122/ActivityRegularizer/LogLog8sequential_122/dense_122/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 22
0sequential_122/dense_122/ActivityRegularizer/Log?
2sequential_122/dense_122/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<24
2sequential_122/dense_122/ActivityRegularizer/mul/x?
0sequential_122/dense_122/ActivityRegularizer/mulMul;sequential_122/dense_122/ActivityRegularizer/mul/x:output:04sequential_122/dense_122/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 22
0sequential_122/dense_122/ActivityRegularizer/mul?
2sequential_122/dense_122/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??24
2sequential_122/dense_122/ActivityRegularizer/sub/x?
0sequential_122/dense_122/ActivityRegularizer/subSub;sequential_122/dense_122/ActivityRegularizer/sub/x:output:08sequential_122/dense_122/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 22
0sequential_122/dense_122/ActivityRegularizer/sub?
8sequential_122/dense_122/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2:
8sequential_122/dense_122/ActivityRegularizer/truediv_1/x?
6sequential_122/dense_122/ActivityRegularizer/truediv_1RealDivAsequential_122/dense_122/ActivityRegularizer/truediv_1/x:output:04sequential_122/dense_122/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 28
6sequential_122/dense_122/ActivityRegularizer/truediv_1?
2sequential_122/dense_122/ActivityRegularizer/Log_1Log:sequential_122/dense_122/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 24
2sequential_122/dense_122/ActivityRegularizer/Log_1?
4sequential_122/dense_122/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?26
4sequential_122/dense_122/ActivityRegularizer/mul_1/x?
2sequential_122/dense_122/ActivityRegularizer/mul_1Mul=sequential_122/dense_122/ActivityRegularizer/mul_1/x:output:06sequential_122/dense_122/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 24
2sequential_122/dense_122/ActivityRegularizer/mul_1?
0sequential_122/dense_122/ActivityRegularizer/addAddV24sequential_122/dense_122/ActivityRegularizer/mul:z:06sequential_122/dense_122/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 22
0sequential_122/dense_122/ActivityRegularizer/add?
2sequential_122/dense_122/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 24
2sequential_122/dense_122/ActivityRegularizer/Const?
0sequential_122/dense_122/ActivityRegularizer/SumSum4sequential_122/dense_122/ActivityRegularizer/add:z:0;sequential_122/dense_122/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 22
0sequential_122/dense_122/ActivityRegularizer/Sum?
4sequential_122/dense_122/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??26
4sequential_122/dense_122/ActivityRegularizer/mul_2/x?
2sequential_122/dense_122/ActivityRegularizer/mul_2Mul=sequential_122/dense_122/ActivityRegularizer/mul_2/x:output:09sequential_122/dense_122/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 24
2sequential_122/dense_122/ActivityRegularizer/mul_2?
2sequential_122/dense_122/ActivityRegularizer/ShapeShape$sequential_122/dense_122/Sigmoid:y:0*
T0*
_output_shapes
:24
2sequential_122/dense_122/ActivityRegularizer/Shape?
@sequential_122/dense_122/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential_122/dense_122/ActivityRegularizer/strided_slice/stack?
Bsequential_122/dense_122/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_122/dense_122/ActivityRegularizer/strided_slice/stack_1?
Bsequential_122/dense_122/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_122/dense_122/ActivityRegularizer/strided_slice/stack_2?
:sequential_122/dense_122/ActivityRegularizer/strided_sliceStridedSlice;sequential_122/dense_122/ActivityRegularizer/Shape:output:0Isequential_122/dense_122/ActivityRegularizer/strided_slice/stack:output:0Ksequential_122/dense_122/ActivityRegularizer/strided_slice/stack_1:output:0Ksequential_122/dense_122/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_122/dense_122/ActivityRegularizer/strided_slice?
1sequential_122/dense_122/ActivityRegularizer/CastCastCsequential_122/dense_122/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 23
1sequential_122/dense_122/ActivityRegularizer/Cast?
6sequential_122/dense_122/ActivityRegularizer/truediv_2RealDiv6sequential_122/dense_122/ActivityRegularizer/mul_2:z:05sequential_122/dense_122/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 28
6sequential_122/dense_122/ActivityRegularizer/truediv_2?
.sequential_123/dense_123/MatMul/ReadVariableOpReadVariableOp7sequential_123_dense_123_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype020
.sequential_123/dense_123/MatMul/ReadVariableOp?
sequential_123/dense_123/MatMulMatMul$sequential_122/dense_122/Sigmoid:y:06sequential_123/dense_123/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2!
sequential_123/dense_123/MatMul?
/sequential_123/dense_123/BiasAdd/ReadVariableOpReadVariableOp8sequential_123_dense_123_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype021
/sequential_123/dense_123/BiasAdd/ReadVariableOp?
 sequential_123/dense_123/BiasAddBiasAdd)sequential_123/dense_123/MatMul:product:07sequential_123/dense_123/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2"
 sequential_123/dense_123/BiasAdd?
 sequential_123/dense_123/SigmoidSigmoid)sequential_123/dense_123/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2"
 sequential_123/dense_123/Sigmoid?
2dense_122/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7sequential_122_dense_122_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_122/kernel/Regularizer/Square/ReadVariableOp?
#dense_122/kernel/Regularizer/SquareSquare:dense_122/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_122/kernel/Regularizer/Square?
"dense_122/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_122/kernel/Regularizer/Const?
 dense_122/kernel/Regularizer/SumSum'dense_122/kernel/Regularizer/Square:y:0+dense_122/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/Sum?
"dense_122/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_122/kernel/Regularizer/mul/x?
 dense_122/kernel/Regularizer/mulMul+dense_122/kernel/Regularizer/mul/x:output:0)dense_122/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/mul?
2dense_123/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7sequential_123_dense_123_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_123/kernel/Regularizer/Square/ReadVariableOp?
#dense_123/kernel/Regularizer/SquareSquare:dense_123/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_123/kernel/Regularizer/Square?
"dense_123/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_123/kernel/Regularizer/Const?
 dense_123/kernel/Regularizer/SumSum'dense_123/kernel/Regularizer/Square:y:0+dense_123/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/Sum?
"dense_123/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_123/kernel/Regularizer/mul/x?
 dense_123/kernel/Regularizer/mulMul+dense_123/kernel/Regularizer/mul/x:output:0)dense_123/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/mul?
IdentityIdentity$sequential_123/dense_123/Sigmoid:y:03^dense_122/kernel/Regularizer/Square/ReadVariableOp3^dense_123/kernel/Regularizer/Square/ReadVariableOp0^sequential_122/dense_122/BiasAdd/ReadVariableOp/^sequential_122/dense_122/MatMul/ReadVariableOp0^sequential_123/dense_123/BiasAdd/ReadVariableOp/^sequential_123/dense_123/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity:sequential_122/dense_122/ActivityRegularizer/truediv_2:z:03^dense_122/kernel/Regularizer/Square/ReadVariableOp3^dense_123/kernel/Regularizer/Square/ReadVariableOp0^sequential_122/dense_122/BiasAdd/ReadVariableOp/^sequential_122/dense_122/MatMul/ReadVariableOp0^sequential_123/dense_123/BiasAdd/ReadVariableOp/^sequential_123/dense_123/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_122/kernel/Regularizer/Square/ReadVariableOp2dense_122/kernel/Regularizer/Square/ReadVariableOp2h
2dense_123/kernel/Regularizer/Square/ReadVariableOp2dense_123/kernel/Regularizer/Square/ReadVariableOp2b
/sequential_122/dense_122/BiasAdd/ReadVariableOp/sequential_122/dense_122/BiasAdd/ReadVariableOp2`
.sequential_122/dense_122/MatMul/ReadVariableOp.sequential_122/dense_122/MatMul/ReadVariableOp2b
/sequential_123/dense_123/BiasAdd/ReadVariableOp/sequential_123/dense_123/BiasAdd/ReadVariableOp2`
.sequential_123/dense_123/MatMul/ReadVariableOp.sequential_123/dense_123/MatMul/ReadVariableOp:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?
?
&__inference_signature_wrapper_16652528
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
#__inference__wrapped_model_166519982
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
__inference_loss_fn_1_16652982M
;dense_123_kernel_regularizer_square_readvariableop_resource: ^
identity??2dense_123/kernel/Regularizer/Square/ReadVariableOp?
2dense_123/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;dense_123_kernel_regularizer_square_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_123/kernel/Regularizer/Square/ReadVariableOp?
#dense_123/kernel/Regularizer/SquareSquare:dense_123/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_123/kernel/Regularizer/Square?
"dense_123/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_123/kernel/Regularizer/Const?
 dense_123/kernel/Regularizer/SumSum'dense_123/kernel/Regularizer/Square:y:0+dense_123/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/Sum?
"dense_123/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_123/kernel/Regularizer/mul/x?
 dense_123/kernel/Regularizer/mulMul+dense_123/kernel/Regularizer/mul/x:output:0)dense_123/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/mul?
IdentityIdentity$dense_123/kernel/Regularizer/mul:z:03^dense_123/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2dense_123/kernel/Regularizer/Square/ReadVariableOp2dense_123/kernel/Regularizer/Square/ReadVariableOp
?
?
L__inference_sequential_123_layer_call_and_return_conditional_losses_16652868

inputs:
(dense_123_matmul_readvariableop_resource: ^7
)dense_123_biasadd_readvariableop_resource:^
identity?? dense_123/BiasAdd/ReadVariableOp?dense_123/MatMul/ReadVariableOp?2dense_123/kernel/Regularizer/Square/ReadVariableOp?
dense_123/MatMul/ReadVariableOpReadVariableOp(dense_123_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02!
dense_123/MatMul/ReadVariableOp?
dense_123/MatMulMatMulinputs'dense_123/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_123/MatMul?
 dense_123/BiasAdd/ReadVariableOpReadVariableOp)dense_123_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02"
 dense_123/BiasAdd/ReadVariableOp?
dense_123/BiasAddBiasAdddense_123/MatMul:product:0(dense_123/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_123/BiasAdd
dense_123/SigmoidSigmoiddense_123/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2
dense_123/Sigmoid?
2dense_123/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_123_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_123/kernel/Regularizer/Square/ReadVariableOp?
#dense_123/kernel/Regularizer/SquareSquare:dense_123/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_123/kernel/Regularizer/Square?
"dense_123/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_123/kernel/Regularizer/Const?
 dense_123/kernel/Regularizer/SumSum'dense_123/kernel/Regularizer/Square:y:0+dense_123/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/Sum?
"dense_123/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_123/kernel/Regularizer/mul/x?
 dense_123/kernel/Regularizer/mulMul+dense_123/kernel/Regularizer/mul/x:output:0)dense_123/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/mul?
IdentityIdentitydense_123/Sigmoid:y:0!^dense_123/BiasAdd/ReadVariableOp ^dense_123/MatMul/ReadVariableOp3^dense_123/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2D
 dense_123/BiasAdd/ReadVariableOp dense_123/BiasAdd/ReadVariableOp2B
dense_123/MatMul/ReadVariableOpdense_123/MatMul/ReadVariableOp2h
2dense_123/kernel/Regularizer/Square/ReadVariableOp2dense_123/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?%
?
L__inference_autoencoder_61_layer_call_and_return_conditional_losses_16652501
input_1)
sequential_122_16652476:^ %
sequential_122_16652478: )
sequential_123_16652482: ^%
sequential_123_16652484:^
identity

identity_1??2dense_122/kernel/Regularizer/Square/ReadVariableOp?2dense_123/kernel/Regularizer/Square/ReadVariableOp?&sequential_122/StatefulPartitionedCall?&sequential_123/StatefulPartitionedCall?
&sequential_122/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_122_16652476sequential_122_16652478*
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
L__inference_sequential_122_layer_call_and_return_conditional_losses_166521392(
&sequential_122/StatefulPartitionedCall?
&sequential_123/StatefulPartitionedCallStatefulPartitionedCall/sequential_122/StatefulPartitionedCall:output:0sequential_123_16652482sequential_123_16652484*
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
L__inference_sequential_123_layer_call_and_return_conditional_losses_166522852(
&sequential_123/StatefulPartitionedCall?
2dense_122/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_122_16652476*
_output_shapes

:^ *
dtype024
2dense_122/kernel/Regularizer/Square/ReadVariableOp?
#dense_122/kernel/Regularizer/SquareSquare:dense_122/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_122/kernel/Regularizer/Square?
"dense_122/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_122/kernel/Regularizer/Const?
 dense_122/kernel/Regularizer/SumSum'dense_122/kernel/Regularizer/Square:y:0+dense_122/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/Sum?
"dense_122/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_122/kernel/Regularizer/mul/x?
 dense_122/kernel/Regularizer/mulMul+dense_122/kernel/Regularizer/mul/x:output:0)dense_122/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/mul?
2dense_123/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_123_16652482*
_output_shapes

: ^*
dtype024
2dense_123/kernel/Regularizer/Square/ReadVariableOp?
#dense_123/kernel/Regularizer/SquareSquare:dense_123/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_123/kernel/Regularizer/Square?
"dense_123/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_123/kernel/Regularizer/Const?
 dense_123/kernel/Regularizer/SumSum'dense_123/kernel/Regularizer/Square:y:0+dense_123/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/Sum?
"dense_123/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_123/kernel/Regularizer/mul/x?
 dense_123/kernel/Regularizer/mulMul+dense_123/kernel/Regularizer/mul/x:output:0)dense_123/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/mul?
IdentityIdentity/sequential_123/StatefulPartitionedCall:output:03^dense_122/kernel/Regularizer/Square/ReadVariableOp3^dense_123/kernel/Regularizer/Square/ReadVariableOp'^sequential_122/StatefulPartitionedCall'^sequential_123/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity/sequential_122/StatefulPartitionedCall:output:13^dense_122/kernel/Regularizer/Square/ReadVariableOp3^dense_123/kernel/Regularizer/Square/ReadVariableOp'^sequential_122/StatefulPartitionedCall'^sequential_123/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_122/kernel/Regularizer/Square/ReadVariableOp2dense_122/kernel/Regularizer/Square/ReadVariableOp2h
2dense_123/kernel/Regularizer/Square/ReadVariableOp2dense_123/kernel/Regularizer/Square/ReadVariableOp2P
&sequential_122/StatefulPartitionedCall&sequential_122/StatefulPartitionedCall2P
&sequential_123/StatefulPartitionedCall&sequential_123/StatefulPartitionedCall:P L
'
_output_shapes
:?????????^
!
_user_specified_name	input_1
?
?
1__inference_sequential_122_layer_call_fn_16652157
input_62
unknown:^ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_62unknown	unknown_0*
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
L__inference_sequential_122_layer_call_and_return_conditional_losses_166521392
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
input_62
?
?
1__inference_sequential_122_layer_call_fn_16652700

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
L__inference_sequential_122_layer_call_and_return_conditional_losses_166521392
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
?
?
1__inference_autoencoder_61_layer_call_fn_16652375
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
L__inference_autoencoder_61_layer_call_and_return_conditional_losses_166523632
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
?
?
G__inference_dense_122_layer_call_and_return_conditional_losses_16652051

inputs0
matmul_readvariableop_resource:^ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2dense_122/kernel/Regularizer/Square/ReadVariableOp?
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
2dense_122/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_122/kernel/Regularizer/Square/ReadVariableOp?
#dense_122/kernel/Regularizer/SquareSquare:dense_122/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_122/kernel/Regularizer/Square?
"dense_122/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_122/kernel/Regularizer/Const?
 dense_122/kernel/Regularizer/SumSum'dense_122/kernel/Regularizer/Square:y:0+dense_122/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/Sum?
"dense_122/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_122/kernel/Regularizer/mul/x?
 dense_122/kernel/Regularizer/mulMul+dense_122/kernel/Regularizer/mul/x:output:0)dense_122/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/mul?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^dense_122/kernel/Regularizer/Square/ReadVariableOp*
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
2dense_122/kernel/Regularizer/Square/ReadVariableOp2dense_122/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?%
?
L__inference_autoencoder_61_layer_call_and_return_conditional_losses_16652473
input_1)
sequential_122_16652448:^ %
sequential_122_16652450: )
sequential_123_16652454: ^%
sequential_123_16652456:^
identity

identity_1??2dense_122/kernel/Regularizer/Square/ReadVariableOp?2dense_123/kernel/Regularizer/Square/ReadVariableOp?&sequential_122/StatefulPartitionedCall?&sequential_123/StatefulPartitionedCall?
&sequential_122/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_122_16652448sequential_122_16652450*
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
L__inference_sequential_122_layer_call_and_return_conditional_losses_166520732(
&sequential_122/StatefulPartitionedCall?
&sequential_123/StatefulPartitionedCallStatefulPartitionedCall/sequential_122/StatefulPartitionedCall:output:0sequential_123_16652454sequential_123_16652456*
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
L__inference_sequential_123_layer_call_and_return_conditional_losses_166522422(
&sequential_123/StatefulPartitionedCall?
2dense_122/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_122_16652448*
_output_shapes

:^ *
dtype024
2dense_122/kernel/Regularizer/Square/ReadVariableOp?
#dense_122/kernel/Regularizer/SquareSquare:dense_122/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_122/kernel/Regularizer/Square?
"dense_122/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_122/kernel/Regularizer/Const?
 dense_122/kernel/Regularizer/SumSum'dense_122/kernel/Regularizer/Square:y:0+dense_122/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/Sum?
"dense_122/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_122/kernel/Regularizer/mul/x?
 dense_122/kernel/Regularizer/mulMul+dense_122/kernel/Regularizer/mul/x:output:0)dense_122/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/mul?
2dense_123/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_123_16652454*
_output_shapes

: ^*
dtype024
2dense_123/kernel/Regularizer/Square/ReadVariableOp?
#dense_123/kernel/Regularizer/SquareSquare:dense_123/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_123/kernel/Regularizer/Square?
"dense_123/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_123/kernel/Regularizer/Const?
 dense_123/kernel/Regularizer/SumSum'dense_123/kernel/Regularizer/Square:y:0+dense_123/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/Sum?
"dense_123/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_123/kernel/Regularizer/mul/x?
 dense_123/kernel/Regularizer/mulMul+dense_123/kernel/Regularizer/mul/x:output:0)dense_123/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/mul?
IdentityIdentity/sequential_123/StatefulPartitionedCall:output:03^dense_122/kernel/Regularizer/Square/ReadVariableOp3^dense_123/kernel/Regularizer/Square/ReadVariableOp'^sequential_122/StatefulPartitionedCall'^sequential_123/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity/sequential_122/StatefulPartitionedCall:output:13^dense_122/kernel/Regularizer/Square/ReadVariableOp3^dense_123/kernel/Regularizer/Square/ReadVariableOp'^sequential_122/StatefulPartitionedCall'^sequential_123/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_122/kernel/Regularizer/Square/ReadVariableOp2dense_122/kernel/Regularizer/Square/ReadVariableOp2h
2dense_123/kernel/Regularizer/Square/ReadVariableOp2dense_123/kernel/Regularizer/Square/ReadVariableOp2P
&sequential_122/StatefulPartitionedCall&sequential_122/StatefulPartitionedCall2P
&sequential_123/StatefulPartitionedCall&sequential_123/StatefulPartitionedCall:P L
'
_output_shapes
:?????????^
!
_user_specified_name	input_1
?
?
1__inference_sequential_123_layer_call_fn_16652825

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
L__inference_sequential_123_layer_call_and_return_conditional_losses_166522852
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
L__inference_sequential_123_layer_call_and_return_conditional_losses_16652851

inputs:
(dense_123_matmul_readvariableop_resource: ^7
)dense_123_biasadd_readvariableop_resource:^
identity?? dense_123/BiasAdd/ReadVariableOp?dense_123/MatMul/ReadVariableOp?2dense_123/kernel/Regularizer/Square/ReadVariableOp?
dense_123/MatMul/ReadVariableOpReadVariableOp(dense_123_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02!
dense_123/MatMul/ReadVariableOp?
dense_123/MatMulMatMulinputs'dense_123/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_123/MatMul?
 dense_123/BiasAdd/ReadVariableOpReadVariableOp)dense_123_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02"
 dense_123/BiasAdd/ReadVariableOp?
dense_123/BiasAddBiasAdddense_123/MatMul:product:0(dense_123/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^2
dense_123/BiasAdd
dense_123/SigmoidSigmoiddense_123/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^2
dense_123/Sigmoid?
2dense_123/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_123_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype024
2dense_123/kernel/Regularizer/Square/ReadVariableOp?
#dense_123/kernel/Regularizer/SquareSquare:dense_123/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_123/kernel/Regularizer/Square?
"dense_123/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_123/kernel/Regularizer/Const?
 dense_123/kernel/Regularizer/SumSum'dense_123/kernel/Regularizer/Square:y:0+dense_123/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/Sum?
"dense_123/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_123/kernel/Regularizer/mul/x?
 dense_123/kernel/Regularizer/mulMul+dense_123/kernel/Regularizer/mul/x:output:0)dense_123/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/mul?
IdentityIdentitydense_123/Sigmoid:y:0!^dense_123/BiasAdd/ReadVariableOp ^dense_123/MatMul/ReadVariableOp3^dense_123/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2D
 dense_123/BiasAdd/ReadVariableOp dense_123/BiasAdd/ReadVariableOp2B
dense_123/MatMul/ReadVariableOpdense_123/MatMul/ReadVariableOp2h
2dense_123/kernel/Regularizer/Square/ReadVariableOp2dense_123/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
L__inference_sequential_123_layer_call_and_return_conditional_losses_16652285

inputs$
dense_123_16652273: ^ 
dense_123_16652275:^
identity??!dense_123/StatefulPartitionedCall?2dense_123/kernel/Regularizer/Square/ReadVariableOp?
!dense_123/StatefulPartitionedCallStatefulPartitionedCallinputsdense_123_16652273dense_123_16652275*
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
G__inference_dense_123_layer_call_and_return_conditional_losses_166522292#
!dense_123/StatefulPartitionedCall?
2dense_123/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_123_16652273*
_output_shapes

: ^*
dtype024
2dense_123/kernel/Regularizer/Square/ReadVariableOp?
#dense_123/kernel/Regularizer/SquareSquare:dense_123/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_123/kernel/Regularizer/Square?
"dense_123/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_123/kernel/Regularizer/Const?
 dense_123/kernel/Regularizer/SumSum'dense_123/kernel/Regularizer/Square:y:0+dense_123/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/Sum?
"dense_123/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_123/kernel/Regularizer/mul/x?
 dense_123/kernel/Regularizer/mulMul+dense_123/kernel/Regularizer/mul/x:output:0)dense_123/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/mul?
IdentityIdentity*dense_123/StatefulPartitionedCall:output:0"^dense_123/StatefulPartitionedCall3^dense_123/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 2F
!dense_123/StatefulPartitionedCall!dense_123/StatefulPartitionedCall2h
2dense_123/kernel/Regularizer/Square/ReadVariableOp2dense_123/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?%
?
L__inference_autoencoder_61_layer_call_and_return_conditional_losses_16652363
x)
sequential_122_16652338:^ %
sequential_122_16652340: )
sequential_123_16652344: ^%
sequential_123_16652346:^
identity

identity_1??2dense_122/kernel/Regularizer/Square/ReadVariableOp?2dense_123/kernel/Regularizer/Square/ReadVariableOp?&sequential_122/StatefulPartitionedCall?&sequential_123/StatefulPartitionedCall?
&sequential_122/StatefulPartitionedCallStatefulPartitionedCallxsequential_122_16652338sequential_122_16652340*
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
L__inference_sequential_122_layer_call_and_return_conditional_losses_166520732(
&sequential_122/StatefulPartitionedCall?
&sequential_123/StatefulPartitionedCallStatefulPartitionedCall/sequential_122/StatefulPartitionedCall:output:0sequential_123_16652344sequential_123_16652346*
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
L__inference_sequential_123_layer_call_and_return_conditional_losses_166522422(
&sequential_123/StatefulPartitionedCall?
2dense_122/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_122_16652338*
_output_shapes

:^ *
dtype024
2dense_122/kernel/Regularizer/Square/ReadVariableOp?
#dense_122/kernel/Regularizer/SquareSquare:dense_122/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_122/kernel/Regularizer/Square?
"dense_122/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_122/kernel/Regularizer/Const?
 dense_122/kernel/Regularizer/SumSum'dense_122/kernel/Regularizer/Square:y:0+dense_122/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/Sum?
"dense_122/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_122/kernel/Regularizer/mul/x?
 dense_122/kernel/Regularizer/mulMul+dense_122/kernel/Regularizer/mul/x:output:0)dense_122/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/mul?
2dense_123/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_123_16652344*
_output_shapes

: ^*
dtype024
2dense_123/kernel/Regularizer/Square/ReadVariableOp?
#dense_123/kernel/Regularizer/SquareSquare:dense_123/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

: ^2%
#dense_123/kernel/Regularizer/Square?
"dense_123/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_123/kernel/Regularizer/Const?
 dense_123/kernel/Regularizer/SumSum'dense_123/kernel/Regularizer/Square:y:0+dense_123/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/Sum?
"dense_123/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_123/kernel/Regularizer/mul/x?
 dense_123/kernel/Regularizer/mulMul+dense_123/kernel/Regularizer/mul/x:output:0)dense_123/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_123/kernel/Regularizer/mul?
IdentityIdentity/sequential_123/StatefulPartitionedCall:output:03^dense_122/kernel/Regularizer/Square/ReadVariableOp3^dense_123/kernel/Regularizer/Square/ReadVariableOp'^sequential_122/StatefulPartitionedCall'^sequential_123/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????^2

Identity?

Identity_1Identity/sequential_122/StatefulPartitionedCall:output:13^dense_122/kernel/Regularizer/Square/ReadVariableOp3^dense_123/kernel/Regularizer/Square/ReadVariableOp'^sequential_122/StatefulPartitionedCall'^sequential_123/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2h
2dense_122/kernel/Regularizer/Square/ReadVariableOp2dense_122/kernel/Regularizer/Square/ReadVariableOp2h
2dense_123/kernel/Regularizer/Square/ReadVariableOp2dense_123/kernel/Regularizer/Square/ReadVariableOp2P
&sequential_122/StatefulPartitionedCall&sequential_122/StatefulPartitionedCall2P
&sequential_123/StatefulPartitionedCall&sequential_123/StatefulPartitionedCall:J F
'
_output_shapes
:?????????^

_user_specified_nameX
?#
?
L__inference_sequential_122_layer_call_and_return_conditional_losses_16652073

inputs$
dense_122_16652052:^  
dense_122_16652054: 
identity

identity_1??!dense_122/StatefulPartitionedCall?2dense_122/kernel/Regularizer/Square/ReadVariableOp?
!dense_122/StatefulPartitionedCallStatefulPartitionedCallinputsdense_122_16652052dense_122_16652054*
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
G__inference_dense_122_layer_call_and_return_conditional_losses_166520512#
!dense_122/StatefulPartitionedCall?
-dense_122/ActivityRegularizer/PartitionedCallPartitionedCall*dense_122/StatefulPartitionedCall:output:0*
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
3__inference_dense_122_activity_regularizer_166520272/
-dense_122/ActivityRegularizer/PartitionedCall?
#dense_122/ActivityRegularizer/ShapeShape*dense_122/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2%
#dense_122/ActivityRegularizer/Shape?
1dense_122/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_122/ActivityRegularizer/strided_slice/stack?
3dense_122/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_122/ActivityRegularizer/strided_slice/stack_1?
3dense_122/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_122/ActivityRegularizer/strided_slice/stack_2?
+dense_122/ActivityRegularizer/strided_sliceStridedSlice,dense_122/ActivityRegularizer/Shape:output:0:dense_122/ActivityRegularizer/strided_slice/stack:output:0<dense_122/ActivityRegularizer/strided_slice/stack_1:output:0<dense_122/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_122/ActivityRegularizer/strided_slice?
"dense_122/ActivityRegularizer/CastCast4dense_122/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_122/ActivityRegularizer/Cast?
%dense_122/ActivityRegularizer/truedivRealDiv6dense_122/ActivityRegularizer/PartitionedCall:output:0&dense_122/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2'
%dense_122/ActivityRegularizer/truediv?
2dense_122/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_122_16652052*
_output_shapes

:^ *
dtype024
2dense_122/kernel/Regularizer/Square/ReadVariableOp?
#dense_122/kernel/Regularizer/SquareSquare:dense_122/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_122/kernel/Regularizer/Square?
"dense_122/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_122/kernel/Regularizer/Const?
 dense_122/kernel/Regularizer/SumSum'dense_122/kernel/Regularizer/Square:y:0+dense_122/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/Sum?
"dense_122/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_122/kernel/Regularizer/mul/x?
 dense_122/kernel/Regularizer/mulMul+dense_122/kernel/Regularizer/mul/x:output:0)dense_122/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/mul?
IdentityIdentity*dense_122/StatefulPartitionedCall:output:0"^dense_122/StatefulPartitionedCall3^dense_122/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity)dense_122/ActivityRegularizer/truediv:z:0"^dense_122/StatefulPartitionedCall3^dense_122/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2F
!dense_122/StatefulPartitionedCall!dense_122/StatefulPartitionedCall2h
2dense_122/kernel/Regularizer/Square/ReadVariableOp2dense_122/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
$__inference__traced_restore_16653056
file_prefix3
!assignvariableop_dense_122_kernel:^ /
!assignvariableop_1_dense_122_bias: 5
#assignvariableop_2_dense_123_kernel: ^/
!assignvariableop_3_dense_123_bias:^

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
AssignVariableOpAssignVariableOp!assignvariableop_dense_122_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_122_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_123_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_123_biasIdentity_3:output:0"/device:CPU:0*
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
?B
?
L__inference_sequential_122_layer_call_and_return_conditional_losses_16652746

inputs:
(dense_122_matmul_readvariableop_resource:^ 7
)dense_122_biasadd_readvariableop_resource: 
identity

identity_1?? dense_122/BiasAdd/ReadVariableOp?dense_122/MatMul/ReadVariableOp?2dense_122/kernel/Regularizer/Square/ReadVariableOp?
dense_122/MatMul/ReadVariableOpReadVariableOp(dense_122_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02!
dense_122/MatMul/ReadVariableOp?
dense_122/MatMulMatMulinputs'dense_122/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_122/MatMul?
 dense_122/BiasAdd/ReadVariableOpReadVariableOp)dense_122_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 dense_122/BiasAdd/ReadVariableOp?
dense_122/BiasAddBiasAdddense_122/MatMul:product:0(dense_122/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_122/BiasAdd
dense_122/SigmoidSigmoiddense_122/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_122/Sigmoid?
4dense_122/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 26
4dense_122/ActivityRegularizer/Mean/reduction_indices?
"dense_122/ActivityRegularizer/MeanMeandense_122/Sigmoid:y:0=dense_122/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2$
"dense_122/ActivityRegularizer/Mean?
'dense_122/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2)
'dense_122/ActivityRegularizer/Maximum/y?
%dense_122/ActivityRegularizer/MaximumMaximum+dense_122/ActivityRegularizer/Mean:output:00dense_122/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2'
%dense_122/ActivityRegularizer/Maximum?
'dense_122/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2)
'dense_122/ActivityRegularizer/truediv/x?
%dense_122/ActivityRegularizer/truedivRealDiv0dense_122/ActivityRegularizer/truediv/x:output:0)dense_122/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2'
%dense_122/ActivityRegularizer/truediv?
!dense_122/ActivityRegularizer/LogLog)dense_122/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2#
!dense_122/ActivityRegularizer/Log?
#dense_122/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_122/ActivityRegularizer/mul/x?
!dense_122/ActivityRegularizer/mulMul,dense_122/ActivityRegularizer/mul/x:output:0%dense_122/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2#
!dense_122/ActivityRegularizer/mul?
#dense_122/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#dense_122/ActivityRegularizer/sub/x?
!dense_122/ActivityRegularizer/subSub,dense_122/ActivityRegularizer/sub/x:output:0)dense_122/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2#
!dense_122/ActivityRegularizer/sub?
)dense_122/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2+
)dense_122/ActivityRegularizer/truediv_1/x?
'dense_122/ActivityRegularizer/truediv_1RealDiv2dense_122/ActivityRegularizer/truediv_1/x:output:0%dense_122/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2)
'dense_122/ActivityRegularizer/truediv_1?
#dense_122/ActivityRegularizer/Log_1Log+dense_122/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2%
#dense_122/ActivityRegularizer/Log_1?
%dense_122/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2'
%dense_122/ActivityRegularizer/mul_1/x?
#dense_122/ActivityRegularizer/mul_1Mul.dense_122/ActivityRegularizer/mul_1/x:output:0'dense_122/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2%
#dense_122/ActivityRegularizer/mul_1?
!dense_122/ActivityRegularizer/addAddV2%dense_122/ActivityRegularizer/mul:z:0'dense_122/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_122/ActivityRegularizer/add?
#dense_122/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#dense_122/ActivityRegularizer/Const?
!dense_122/ActivityRegularizer/SumSum%dense_122/ActivityRegularizer/add:z:0,dense_122/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2#
!dense_122/ActivityRegularizer/Sum?
%dense_122/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%dense_122/ActivityRegularizer/mul_2/x?
#dense_122/ActivityRegularizer/mul_2Mul.dense_122/ActivityRegularizer/mul_2/x:output:0*dense_122/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#dense_122/ActivityRegularizer/mul_2?
#dense_122/ActivityRegularizer/ShapeShapedense_122/Sigmoid:y:0*
T0*
_output_shapes
:2%
#dense_122/ActivityRegularizer/Shape?
1dense_122/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1dense_122/ActivityRegularizer/strided_slice/stack?
3dense_122/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_122/ActivityRegularizer/strided_slice/stack_1?
3dense_122/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3dense_122/ActivityRegularizer/strided_slice/stack_2?
+dense_122/ActivityRegularizer/strided_sliceStridedSlice,dense_122/ActivityRegularizer/Shape:output:0:dense_122/ActivityRegularizer/strided_slice/stack:output:0<dense_122/ActivityRegularizer/strided_slice/stack_1:output:0<dense_122/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+dense_122/ActivityRegularizer/strided_slice?
"dense_122/ActivityRegularizer/CastCast4dense_122/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"dense_122/ActivityRegularizer/Cast?
'dense_122/ActivityRegularizer/truediv_2RealDiv'dense_122/ActivityRegularizer/mul_2:z:0&dense_122/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2)
'dense_122/ActivityRegularizer/truediv_2?
2dense_122/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(dense_122_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype024
2dense_122/kernel/Regularizer/Square/ReadVariableOp?
#dense_122/kernel/Regularizer/SquareSquare:dense_122/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:^ 2%
#dense_122/kernel/Regularizer/Square?
"dense_122/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_122/kernel/Regularizer/Const?
 dense_122/kernel/Regularizer/SumSum'dense_122/kernel/Regularizer/Square:y:0+dense_122/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/Sum?
"dense_122/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_122/kernel/Regularizer/mul/x?
 dense_122/kernel/Regularizer/mulMul+dense_122/kernel/Regularizer/mul/x:output:0)dense_122/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 dense_122/kernel/Regularizer/mul?
IdentityIdentitydense_122/Sigmoid:y:0!^dense_122/BiasAdd/ReadVariableOp ^dense_122/MatMul/ReadVariableOp3^dense_122/kernel/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity+dense_122/ActivityRegularizer/truediv_2:z:0!^dense_122/BiasAdd/ReadVariableOp ^dense_122/MatMul/ReadVariableOp3^dense_122/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????^: : 2D
 dense_122/BiasAdd/ReadVariableOp dense_122/BiasAdd/ReadVariableOp2B
dense_122/MatMul/ReadVariableOpdense_122/MatMul/ReadVariableOp2h
2dense_122/kernel/Regularizer/Square/ReadVariableOp2dense_122/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????^
 
_user_specified_nameinputs
?
?
1__inference_sequential_123_layer_call_fn_16652834
dense_123_input
unknown: ^
	unknown_0:^
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_123_inputunknown	unknown_0*
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
L__inference_sequential_123_layer_call_and_return_conditional_losses_166522852
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
_user_specified_namedense_123_input
?_
?
#__inference__wrapped_model_16651998
input_1X
Fautoencoder_61_sequential_122_dense_122_matmul_readvariableop_resource:^ U
Gautoencoder_61_sequential_122_dense_122_biasadd_readvariableop_resource: X
Fautoencoder_61_sequential_123_dense_123_matmul_readvariableop_resource: ^U
Gautoencoder_61_sequential_123_dense_123_biasadd_readvariableop_resource:^
identity??>autoencoder_61/sequential_122/dense_122/BiasAdd/ReadVariableOp?=autoencoder_61/sequential_122/dense_122/MatMul/ReadVariableOp?>autoencoder_61/sequential_123/dense_123/BiasAdd/ReadVariableOp?=autoencoder_61/sequential_123/dense_123/MatMul/ReadVariableOp?
=autoencoder_61/sequential_122/dense_122/MatMul/ReadVariableOpReadVariableOpFautoencoder_61_sequential_122_dense_122_matmul_readvariableop_resource*
_output_shapes

:^ *
dtype02?
=autoencoder_61/sequential_122/dense_122/MatMul/ReadVariableOp?
.autoencoder_61/sequential_122/dense_122/MatMulMatMulinput_1Eautoencoder_61/sequential_122/dense_122/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 20
.autoencoder_61/sequential_122/dense_122/MatMul?
>autoencoder_61/sequential_122/dense_122/BiasAdd/ReadVariableOpReadVariableOpGautoencoder_61_sequential_122_dense_122_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02@
>autoencoder_61/sequential_122/dense_122/BiasAdd/ReadVariableOp?
/autoencoder_61/sequential_122/dense_122/BiasAddBiasAdd8autoencoder_61/sequential_122/dense_122/MatMul:product:0Fautoencoder_61/sequential_122/dense_122/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 21
/autoencoder_61/sequential_122/dense_122/BiasAdd?
/autoencoder_61/sequential_122/dense_122/SigmoidSigmoid8autoencoder_61/sequential_122/dense_122/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 21
/autoencoder_61/sequential_122/dense_122/Sigmoid?
Rautoencoder_61/sequential_122/dense_122/ActivityRegularizer/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2T
Rautoencoder_61/sequential_122/dense_122/ActivityRegularizer/Mean/reduction_indices?
@autoencoder_61/sequential_122/dense_122/ActivityRegularizer/MeanMean3autoencoder_61/sequential_122/dense_122/Sigmoid:y:0[autoencoder_61/sequential_122/dense_122/ActivityRegularizer/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2B
@autoencoder_61/sequential_122/dense_122/ActivityRegularizer/Mean?
Eautoencoder_61/sequential_122/dense_122/ActivityRegularizer/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???.2G
Eautoencoder_61/sequential_122/dense_122/ActivityRegularizer/Maximum/y?
Cautoencoder_61/sequential_122/dense_122/ActivityRegularizer/MaximumMaximumIautoencoder_61/sequential_122/dense_122/ActivityRegularizer/Mean:output:0Nautoencoder_61/sequential_122/dense_122/ActivityRegularizer/Maximum/y:output:0*
T0*
_output_shapes
: 2E
Cautoencoder_61/sequential_122/dense_122/ActivityRegularizer/Maximum?
Eautoencoder_61/sequential_122/dense_122/ActivityRegularizer/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2G
Eautoencoder_61/sequential_122/dense_122/ActivityRegularizer/truediv/x?
Cautoencoder_61/sequential_122/dense_122/ActivityRegularizer/truedivRealDivNautoencoder_61/sequential_122/dense_122/ActivityRegularizer/truediv/x:output:0Gautoencoder_61/sequential_122/dense_122/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2E
Cautoencoder_61/sequential_122/dense_122/ActivityRegularizer/truediv?
?autoencoder_61/sequential_122/dense_122/ActivityRegularizer/LogLogGautoencoder_61/sequential_122/dense_122/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2A
?autoencoder_61/sequential_122/dense_122/ActivityRegularizer/Log?
Aautoencoder_61/sequential_122/dense_122/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2C
Aautoencoder_61/sequential_122/dense_122/ActivityRegularizer/mul/x?
?autoencoder_61/sequential_122/dense_122/ActivityRegularizer/mulMulJautoencoder_61/sequential_122/dense_122/ActivityRegularizer/mul/x:output:0Cautoencoder_61/sequential_122/dense_122/ActivityRegularizer/Log:y:0*
T0*
_output_shapes
: 2A
?autoencoder_61/sequential_122/dense_122/ActivityRegularizer/mul?
Aautoencoder_61/sequential_122/dense_122/ActivityRegularizer/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2C
Aautoencoder_61/sequential_122/dense_122/ActivityRegularizer/sub/x?
?autoencoder_61/sequential_122/dense_122/ActivityRegularizer/subSubJautoencoder_61/sequential_122/dense_122/ActivityRegularizer/sub/x:output:0Gautoencoder_61/sequential_122/dense_122/ActivityRegularizer/Maximum:z:0*
T0*
_output_shapes
: 2A
?autoencoder_61/sequential_122/dense_122/ActivityRegularizer/sub?
Gautoencoder_61/sequential_122/dense_122/ActivityRegularizer/truediv_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2I
Gautoencoder_61/sequential_122/dense_122/ActivityRegularizer/truediv_1/x?
Eautoencoder_61/sequential_122/dense_122/ActivityRegularizer/truediv_1RealDivPautoencoder_61/sequential_122/dense_122/ActivityRegularizer/truediv_1/x:output:0Cautoencoder_61/sequential_122/dense_122/ActivityRegularizer/sub:z:0*
T0*
_output_shapes
: 2G
Eautoencoder_61/sequential_122/dense_122/ActivityRegularizer/truediv_1?
Aautoencoder_61/sequential_122/dense_122/ActivityRegularizer/Log_1LogIautoencoder_61/sequential_122/dense_122/ActivityRegularizer/truediv_1:z:0*
T0*
_output_shapes
: 2C
Aautoencoder_61/sequential_122/dense_122/ActivityRegularizer/Log_1?
Cautoencoder_61/sequential_122/dense_122/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *?p}?2E
Cautoencoder_61/sequential_122/dense_122/ActivityRegularizer/mul_1/x?
Aautoencoder_61/sequential_122/dense_122/ActivityRegularizer/mul_1MulLautoencoder_61/sequential_122/dense_122/ActivityRegularizer/mul_1/x:output:0Eautoencoder_61/sequential_122/dense_122/ActivityRegularizer/Log_1:y:0*
T0*
_output_shapes
: 2C
Aautoencoder_61/sequential_122/dense_122/ActivityRegularizer/mul_1?
?autoencoder_61/sequential_122/dense_122/ActivityRegularizer/addAddV2Cautoencoder_61/sequential_122/dense_122/ActivityRegularizer/mul:z:0Eautoencoder_61/sequential_122/dense_122/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2A
?autoencoder_61/sequential_122/dense_122/ActivityRegularizer/add?
Aautoencoder_61/sequential_122/dense_122/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Aautoencoder_61/sequential_122/dense_122/ActivityRegularizer/Const?
?autoencoder_61/sequential_122/dense_122/ActivityRegularizer/SumSumCautoencoder_61/sequential_122/dense_122/ActivityRegularizer/add:z:0Jautoencoder_61/sequential_122/dense_122/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2A
?autoencoder_61/sequential_122/dense_122/ActivityRegularizer/Sum?
Cautoencoder_61/sequential_122/dense_122/ActivityRegularizer/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2E
Cautoencoder_61/sequential_122/dense_122/ActivityRegularizer/mul_2/x?
Aautoencoder_61/sequential_122/dense_122/ActivityRegularizer/mul_2MulLautoencoder_61/sequential_122/dense_122/ActivityRegularizer/mul_2/x:output:0Hautoencoder_61/sequential_122/dense_122/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2C
Aautoencoder_61/sequential_122/dense_122/ActivityRegularizer/mul_2?
Aautoencoder_61/sequential_122/dense_122/ActivityRegularizer/ShapeShape3autoencoder_61/sequential_122/dense_122/Sigmoid:y:0*
T0*
_output_shapes
:2C
Aautoencoder_61/sequential_122/dense_122/ActivityRegularizer/Shape?
Oautoencoder_61/sequential_122/dense_122/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2Q
Oautoencoder_61/sequential_122/dense_122/ActivityRegularizer/strided_slice/stack?
Qautoencoder_61/sequential_122/dense_122/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2S
Qautoencoder_61/sequential_122/dense_122/ActivityRegularizer/strided_slice/stack_1?
Qautoencoder_61/sequential_122/dense_122/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2S
Qautoencoder_61/sequential_122/dense_122/ActivityRegularizer/strided_slice/stack_2?
Iautoencoder_61/sequential_122/dense_122/ActivityRegularizer/strided_sliceStridedSliceJautoencoder_61/sequential_122/dense_122/ActivityRegularizer/Shape:output:0Xautoencoder_61/sequential_122/dense_122/ActivityRegularizer/strided_slice/stack:output:0Zautoencoder_61/sequential_122/dense_122/ActivityRegularizer/strided_slice/stack_1:output:0Zautoencoder_61/sequential_122/dense_122/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2K
Iautoencoder_61/sequential_122/dense_122/ActivityRegularizer/strided_slice?
@autoencoder_61/sequential_122/dense_122/ActivityRegularizer/CastCastRautoencoder_61/sequential_122/dense_122/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2B
@autoencoder_61/sequential_122/dense_122/ActivityRegularizer/Cast?
Eautoencoder_61/sequential_122/dense_122/ActivityRegularizer/truediv_2RealDivEautoencoder_61/sequential_122/dense_122/ActivityRegularizer/mul_2:z:0Dautoencoder_61/sequential_122/dense_122/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2G
Eautoencoder_61/sequential_122/dense_122/ActivityRegularizer/truediv_2?
=autoencoder_61/sequential_123/dense_123/MatMul/ReadVariableOpReadVariableOpFautoencoder_61_sequential_123_dense_123_matmul_readvariableop_resource*
_output_shapes

: ^*
dtype02?
=autoencoder_61/sequential_123/dense_123/MatMul/ReadVariableOp?
.autoencoder_61/sequential_123/dense_123/MatMulMatMul3autoencoder_61/sequential_122/dense_122/Sigmoid:y:0Eautoencoder_61/sequential_123/dense_123/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^20
.autoencoder_61/sequential_123/dense_123/MatMul?
>autoencoder_61/sequential_123/dense_123/BiasAdd/ReadVariableOpReadVariableOpGautoencoder_61_sequential_123_dense_123_biasadd_readvariableop_resource*
_output_shapes
:^*
dtype02@
>autoencoder_61/sequential_123/dense_123/BiasAdd/ReadVariableOp?
/autoencoder_61/sequential_123/dense_123/BiasAddBiasAdd8autoencoder_61/sequential_123/dense_123/MatMul:product:0Fautoencoder_61/sequential_123/dense_123/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^21
/autoencoder_61/sequential_123/dense_123/BiasAdd?
/autoencoder_61/sequential_123/dense_123/SigmoidSigmoid8autoencoder_61/sequential_123/dense_123/BiasAdd:output:0*
T0*'
_output_shapes
:?????????^21
/autoencoder_61/sequential_123/dense_123/Sigmoid?
IdentityIdentity3autoencoder_61/sequential_123/dense_123/Sigmoid:y:0?^autoencoder_61/sequential_122/dense_122/BiasAdd/ReadVariableOp>^autoencoder_61/sequential_122/dense_122/MatMul/ReadVariableOp?^autoencoder_61/sequential_123/dense_123/BiasAdd/ReadVariableOp>^autoencoder_61/sequential_123/dense_123/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????^: : : : 2?
>autoencoder_61/sequential_122/dense_122/BiasAdd/ReadVariableOp>autoencoder_61/sequential_122/dense_122/BiasAdd/ReadVariableOp2~
=autoencoder_61/sequential_122/dense_122/MatMul/ReadVariableOp=autoencoder_61/sequential_122/dense_122/MatMul/ReadVariableOp2?
>autoencoder_61/sequential_123/dense_123/BiasAdd/ReadVariableOp>autoencoder_61/sequential_123/dense_123/BiasAdd/ReadVariableOp2~
=autoencoder_61/sequential_123/dense_123/MatMul/ReadVariableOp=autoencoder_61/sequential_123/dense_123/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????^
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
_tf_keras_model?{"name": "autoencoder_61", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Autoencoder", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 94]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Autoencoder"}}
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
_tf_keras_sequential?{"name": "sequential_122", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_122", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 94]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_62"}}, {"class_name": "Dense", "config": {"name": "dense_122", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}}, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 94}}, "shared_object_id": 6}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 94]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 94]}, "float32", "input_62"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_122", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 94]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_62"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_122", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}]}}}
?
layer_with_weights-0
layer-0
trainable_variables
	variables
regularization_losses
	keras_api
=__call__
*>&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"name": "sequential_123", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_123", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_123_input"}}, {"class_name": "Dense", "config": {"name": "dense_123", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [94, 32]}, "float32", "dense_123_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_123", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_123_input"}, "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "dense_123", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8}]}}}
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
{"name": "dense_122", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_122", "trainable": true, "dtype": "float32", "units": 32, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 94}}, "shared_object_id": 6}, "activity_regularizer": {"class_name": "SparseRegularizer", "config": {"rho": 0.01, "beta": 1}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 94]}}
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
_tf_keras_layer?{"name": "dense_123", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_123", "trainable": true, "dtype": "float32", "units": 94, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "RandomUniform", "config": {"minval": 0, "maxval": 0.01, "seed": null}, "shared_object_id": 1}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 2}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [94, 32]}}
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
": ^ 2dense_122/kernel
: 2dense_122/bias
":  ^2dense_123/kernel
:^2dense_123/bias
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
1__inference_autoencoder_61_layer_call_fn_16652375
1__inference_autoencoder_61_layer_call_fn_16652542
1__inference_autoencoder_61_layer_call_fn_16652556
1__inference_autoencoder_61_layer_call_fn_16652445?
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
#__inference__wrapped_model_16651998?
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
L__inference_autoencoder_61_layer_call_and_return_conditional_losses_16652615
L__inference_autoencoder_61_layer_call_and_return_conditional_losses_16652674
L__inference_autoencoder_61_layer_call_and_return_conditional_losses_16652473
L__inference_autoencoder_61_layer_call_and_return_conditional_losses_16652501?
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
1__inference_sequential_122_layer_call_fn_16652081
1__inference_sequential_122_layer_call_fn_16652690
1__inference_sequential_122_layer_call_fn_16652700
1__inference_sequential_122_layer_call_fn_16652157?
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
L__inference_sequential_122_layer_call_and_return_conditional_losses_16652746
L__inference_sequential_122_layer_call_and_return_conditional_losses_16652792
L__inference_sequential_122_layer_call_and_return_conditional_losses_16652181
L__inference_sequential_122_layer_call_and_return_conditional_losses_16652205?
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
1__inference_sequential_123_layer_call_fn_16652807
1__inference_sequential_123_layer_call_fn_16652816
1__inference_sequential_123_layer_call_fn_16652825
1__inference_sequential_123_layer_call_fn_16652834?
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
L__inference_sequential_123_layer_call_and_return_conditional_losses_16652851
L__inference_sequential_123_layer_call_and_return_conditional_losses_16652868
L__inference_sequential_123_layer_call_and_return_conditional_losses_16652885
L__inference_sequential_123_layer_call_and_return_conditional_losses_16652902?
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
&__inference_signature_wrapper_16652528input_1"?
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
,__inference_dense_122_layer_call_fn_16652917?
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
K__inference_dense_122_layer_call_and_return_all_conditional_losses_16652928?
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
__inference_loss_fn_0_16652939?
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
,__inference_dense_123_layer_call_fn_16652954?
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
G__inference_dense_123_layer_call_and_return_conditional_losses_16652971?
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
__inference_loss_fn_1_16652982?
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
3__inference_dense_122_activity_regularizer_16652027?
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
G__inference_dense_122_layer_call_and_return_conditional_losses_16652999?
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
#__inference__wrapped_model_16651998m0?-
&?#
!?
input_1?????????^
? "3?0
.
output_1"?
output_1?????????^?
L__inference_autoencoder_61_layer_call_and_return_conditional_losses_16652473q4?1
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
L__inference_autoencoder_61_layer_call_and_return_conditional_losses_16652501q4?1
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
L__inference_autoencoder_61_layer_call_and_return_conditional_losses_16652615k.?+
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
L__inference_autoencoder_61_layer_call_and_return_conditional_losses_16652674k.?+
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
1__inference_autoencoder_61_layer_call_fn_16652375V4?1
*?'
!?
input_1?????????^
p 
? "??????????^?
1__inference_autoencoder_61_layer_call_fn_16652445V4?1
*?'
!?
input_1?????????^
p
? "??????????^?
1__inference_autoencoder_61_layer_call_fn_16652542P.?+
$?!
?
X?????????^
p 
? "??????????^?
1__inference_autoencoder_61_layer_call_fn_16652556P.?+
$?!
?
X?????????^
p
? "??????????^f
3__inference_dense_122_activity_regularizer_16652027/$?!
?
?

activation
? "? ?
K__inference_dense_122_layer_call_and_return_all_conditional_losses_16652928j/?,
%?"
 ?
inputs?????????^
? "3?0
?
0????????? 
?
?	
1/0 ?
G__inference_dense_122_layer_call_and_return_conditional_losses_16652999\/?,
%?"
 ?
inputs?????????^
? "%?"
?
0????????? 
? 
,__inference_dense_122_layer_call_fn_16652917O/?,
%?"
 ?
inputs?????????^
? "?????????? ?
G__inference_dense_123_layer_call_and_return_conditional_losses_16652971\/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????^
? 
,__inference_dense_123_layer_call_fn_16652954O/?,
%?"
 ?
inputs????????? 
? "??????????^=
__inference_loss_fn_0_16652939?

? 
? "? =
__inference_loss_fn_1_16652982?

? 
? "? ?
L__inference_sequential_122_layer_call_and_return_conditional_losses_16652181t9?6
/?,
"?
input_62?????????^
p 

 
? "3?0
?
0????????? 
?
?	
1/0 ?
L__inference_sequential_122_layer_call_and_return_conditional_losses_16652205t9?6
/?,
"?
input_62?????????^
p

 
? "3?0
?
0????????? 
?
?	
1/0 ?
L__inference_sequential_122_layer_call_and_return_conditional_losses_16652746r7?4
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
L__inference_sequential_122_layer_call_and_return_conditional_losses_16652792r7?4
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
1__inference_sequential_122_layer_call_fn_16652081Y9?6
/?,
"?
input_62?????????^
p 

 
? "?????????? ?
1__inference_sequential_122_layer_call_fn_16652157Y9?6
/?,
"?
input_62?????????^
p

 
? "?????????? ?
1__inference_sequential_122_layer_call_fn_16652690W7?4
-?*
 ?
inputs?????????^
p 

 
? "?????????? ?
1__inference_sequential_122_layer_call_fn_16652700W7?4
-?*
 ?
inputs?????????^
p

 
? "?????????? ?
L__inference_sequential_123_layer_call_and_return_conditional_losses_16652851d7?4
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
L__inference_sequential_123_layer_call_and_return_conditional_losses_16652868d7?4
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
L__inference_sequential_123_layer_call_and_return_conditional_losses_16652885m@?=
6?3
)?&
dense_123_input????????? 
p 

 
? "%?"
?
0?????????^
? ?
L__inference_sequential_123_layer_call_and_return_conditional_losses_16652902m@?=
6?3
)?&
dense_123_input????????? 
p

 
? "%?"
?
0?????????^
? ?
1__inference_sequential_123_layer_call_fn_16652807`@?=
6?3
)?&
dense_123_input????????? 
p 

 
? "??????????^?
1__inference_sequential_123_layer_call_fn_16652816W7?4
-?*
 ?
inputs????????? 
p 

 
? "??????????^?
1__inference_sequential_123_layer_call_fn_16652825W7?4
-?*
 ?
inputs????????? 
p

 
? "??????????^?
1__inference_sequential_123_layer_call_fn_16652834`@?=
6?3
)?&
dense_123_input????????? 
p

 
? "??????????^?
&__inference_signature_wrapper_16652528x;?8
? 
1?.
,
input_1!?
input_1?????????^"3?0
.
output_1"?
output_1?????????^