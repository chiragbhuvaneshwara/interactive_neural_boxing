#
# Autogenerated by Thrift Compiler (0.12.0)
#
# DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
#
#  options string: py
#

from thrift.Thrift import TType, TMessageType, TFrozenDict, TException, TApplicationException
from thrift.protocol.TProtocol import TProtocolException
from thrift.TRecursive import fix_spec

import sys

from thrift.transport import TTransport
all_structs = []


class TGait(object):
    standing = 0
    walking = 1
    running = 2

    _VALUES_TO_NAMES = {
        0: "standing",
        1: "walking",
        2: "running",
    }

    _NAMES_TO_VALUES = {
        "standing": 0,
        "walking": 1,
        "running": 2,
    }


class TPosture(object):
    """
    Attributes:
     - bones
     - bone_map
     - location
     - rotation

    """


    def __init__(self, bones=None, bone_map=None, location=None, rotation=None,):
        self.bones = bones
        self.bone_map = bone_map
        self.location = location
        self.rotation = rotation

    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 1:
                if ftype == TType.LIST:
                    self.bones = []
                    (_etype3, _size0) = iprot.readListBegin()
                    for _i4 in range(_size0):
                        _elem5 = TBone()
                        _elem5.read(iprot)
                        self.bones.append(_elem5)
                    iprot.readListEnd()
                else:
                    iprot.skip(ftype)
            elif fid == 2:
                if ftype == TType.MAP:
                    self.bone_map = {}
                    (_ktype7, _vtype8, _size6) = iprot.readMapBegin()
                    for _i10 in range(_size6):
                        _key11 = iprot.readString().decode('utf-8') if sys.version_info[0] == 2 else iprot.readString()
                        _val12 = iprot.readI32()
                        self.bone_map[_key11] = _val12
                    iprot.readMapEnd()
                else:
                    iprot.skip(ftype)
            elif fid == 3:
                if ftype == TType.STRUCT:
                    self.location = TVector3()
                    self.location.read(iprot)
                else:
                    iprot.skip(ftype)
            elif fid == 4:
                if ftype == TType.DOUBLE:
                    self.rotation = iprot.readDouble()
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('TPosture')
        if self.bones is not None:
            oprot.writeFieldBegin('bones', TType.LIST, 1)
            oprot.writeListBegin(TType.STRUCT, len(self.bones))
            for iter13 in self.bones:
                iter13.write(oprot)
            oprot.writeListEnd()
            oprot.writeFieldEnd()
        if self.bone_map is not None:
            oprot.writeFieldBegin('bone_map', TType.MAP, 2)
            oprot.writeMapBegin(TType.STRING, TType.I32, len(self.bone_map))
            for kiter14, viter15 in self.bone_map.items():
                oprot.writeString(kiter14.encode('utf-8') if sys.version_info[0] == 2 else kiter14)
                oprot.writeI32(viter15)
            oprot.writeMapEnd()
            oprot.writeFieldEnd()
        if self.location is not None:
            oprot.writeFieldBegin('location', TType.STRUCT, 3)
            self.location.write(oprot)
            oprot.writeFieldEnd()
        if self.rotation is not None:
            oprot.writeFieldBegin('rotation', TType.DOUBLE, 4)
            oprot.writeDouble(self.rotation)
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        if self.bones is None:
            raise TProtocolException(message='Required field bones is unset!')
        if self.bone_map is None:
            raise TProtocolException(message='Required field bone_map is unset!')
        if self.location is None:
            raise TProtocolException(message='Required field location is unset!')
        if self.rotation is None:
            raise TProtocolException(message='Required field rotation is unset!')
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)


class TBone(object):
    """
    Attributes:
     - name
     - Position
     - children
     - parent

    """


    def __init__(self, name=None, Position=None, children=None, parent=None,):
        self.name = name
        self.Position = Position
        self.children = children
        self.parent = parent

    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 1:
                if ftype == TType.STRING:
                    self.name = iprot.readString().decode('utf-8') if sys.version_info[0] == 2 else iprot.readString()
                else:
                    iprot.skip(ftype)
            elif fid == 2:
                if ftype == TType.STRUCT:
                    self.Position = TVector3()
                    self.Position.read(iprot)
                else:
                    iprot.skip(ftype)
            elif fid == 3:
                if ftype == TType.LIST:
                    self.children = []
                    (_etype19, _size16) = iprot.readListBegin()
                    for _i20 in range(_size16):
                        _elem21 = iprot.readString().decode('utf-8') if sys.version_info[0] == 2 else iprot.readString()
                        self.children.append(_elem21)
                    iprot.readListEnd()
                else:
                    iprot.skip(ftype)
            elif fid == 4:
                if ftype == TType.STRING:
                    self.parent = iprot.readString().decode('utf-8') if sys.version_info[0] == 2 else iprot.readString()
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('TBone')
        if self.name is not None:
            oprot.writeFieldBegin('name', TType.STRING, 1)
            oprot.writeString(self.name.encode('utf-8') if sys.version_info[0] == 2 else self.name)
            oprot.writeFieldEnd()
        if self.Position is not None:
            oprot.writeFieldBegin('Position', TType.STRUCT, 2)
            self.Position.write(oprot)
            oprot.writeFieldEnd()
        if self.children is not None:
            oprot.writeFieldBegin('children', TType.LIST, 3)
            oprot.writeListBegin(TType.STRING, len(self.children))
            for iter22 in self.children:
                oprot.writeString(iter22.encode('utf-8') if sys.version_info[0] == 2 else iter22)
            oprot.writeListEnd()
            oprot.writeFieldEnd()
        if self.parent is not None:
            oprot.writeFieldBegin('parent', TType.STRING, 4)
            oprot.writeString(self.parent.encode('utf-8') if sys.version_info[0] == 2 else self.parent)
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        if self.name is None:
            raise TProtocolException(message='Required field name is unset!')
        if self.Position is None:
            raise TProtocolException(message='Required field Position is unset!')
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)


class TVector3(object):
    """
    Attributes:
     - X
     - Y
     - Z

    """


    def __init__(self, X=None, Y=None, Z=None,):
        self.X = X
        self.Y = Y
        self.Z = Z

    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, [self.__class__, self.thrift_spec])
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 1:
                if ftype == TType.DOUBLE:
                    self.X = iprot.readDouble()
                else:
                    iprot.skip(ftype)
            elif fid == 2:
                if ftype == TType.DOUBLE:
                    self.Y = iprot.readDouble()
                else:
                    iprot.skip(ftype)
            elif fid == 3:
                if ftype == TType.DOUBLE:
                    self.Z = iprot.readDouble()
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, [self.__class__, self.thrift_spec]))
            return
        oprot.writeStructBegin('TVector3')
        if self.X is not None:
            oprot.writeFieldBegin('X', TType.DOUBLE, 1)
            oprot.writeDouble(self.X)
            oprot.writeFieldEnd()
        if self.Y is not None:
            oprot.writeFieldBegin('Y', TType.DOUBLE, 2)
            oprot.writeDouble(self.Y)
            oprot.writeFieldEnd()
        if self.Z is not None:
            oprot.writeFieldBegin('Z', TType.DOUBLE, 3)
            oprot.writeDouble(self.Z)
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        if self.X is None:
            raise TProtocolException(message='Required field X is unset!')
        if self.Y is None:
            raise TProtocolException(message='Required field Y is unset!')
        if self.Z is None:
            raise TProtocolException(message='Required field Z is unset!')
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in self.__dict__.items()]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)
all_structs.append(TPosture)
TPosture.thrift_spec = (
    None,  # 0
    (1, TType.LIST, 'bones', (TType.STRUCT, [TBone, None], False), None, ),  # 1
    (2, TType.MAP, 'bone_map', (TType.STRING, 'UTF8', TType.I32, None, False), None, ),  # 2
    (3, TType.STRUCT, 'location', [TVector3, None], None, ),  # 3
    (4, TType.DOUBLE, 'rotation', None, None, ),  # 4
)
all_structs.append(TBone)
TBone.thrift_spec = (
    None,  # 0
    (1, TType.STRING, 'name', 'UTF8', None, ),  # 1
    (2, TType.STRUCT, 'Position', [TVector3, None], None, ),  # 2
    (3, TType.LIST, 'children', (TType.STRING, 'UTF8', False), None, ),  # 3
    (4, TType.STRING, 'parent', 'UTF8', None, ),  # 4
)
all_structs.append(TVector3)
TVector3.thrift_spec = (
    None,  # 0
    (1, TType.DOUBLE, 'X', None, None, ),  # 1
    (2, TType.DOUBLE, 'Y', None, None, ),  # 2
    (3, TType.DOUBLE, 'Z', None, None, ),  # 3
)
fix_spec(all_structs)
del all_structs
