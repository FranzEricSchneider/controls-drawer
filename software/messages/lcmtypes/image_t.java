/* LCM type definition class file
 * This file was automatically generated by lcm-gen
 * DO NOT MODIFY BY HAND!!!!
 */

package lcmtypes;
 
import java.io.*;
import java.util.*;
import lcm.lcm.*;
 
public final class image_t implements lcm.lcm.LCMEncodable
{
    public long utime;
    public long action_id;
    public lcmtypes.image_request_t request;
    public short width;
    public short height;
    public short row_stride;
    public short FPS;
    public float brightness;
    public float contrast;
    public int num_data;
    public short data[];
 
    public image_t()
    {
    }
 
    public static final long LCM_FINGERPRINT;
    public static final long LCM_FINGERPRINT_BASE = 0x05a5c6336cf623e9L;
 
    static {
        LCM_FINGERPRINT = _hashRecursive(new ArrayList<Class<?>>());
    }
 
    public static long _hashRecursive(ArrayList<Class<?>> classes)
    {
        if (classes.contains(lcmtypes.image_t.class))
            return 0L;
 
        classes.add(lcmtypes.image_t.class);
        long hash = LCM_FINGERPRINT_BASE
             + lcmtypes.image_request_t._hashRecursive(classes)
            ;
        classes.remove(classes.size() - 1);
        return (hash<<1) + ((hash>>63)&1);
    }
 
    public void encode(DataOutput outs) throws IOException
    {
        outs.writeLong(LCM_FINGERPRINT);
        _encodeRecursive(outs);
    }
 
    public void _encodeRecursive(DataOutput outs) throws IOException
    {
        outs.writeLong(this.utime); 
 
        outs.writeLong(this.action_id); 
 
        this.request._encodeRecursive(outs); 
 
        outs.writeShort(this.width); 
 
        outs.writeShort(this.height); 
 
        outs.writeShort(this.row_stride); 
 
        outs.writeShort(this.FPS); 
 
        outs.writeFloat(this.brightness); 
 
        outs.writeFloat(this.contrast); 
 
        outs.writeInt(this.num_data); 
 
        for (int a = 0; a < this.num_data; a++) {
            outs.writeShort(this.data[a]); 
        }
 
    }
 
    public image_t(byte[] data) throws IOException
    {
        this(new LCMDataInputStream(data));
    }
 
    public image_t(DataInput ins) throws IOException
    {
        if (ins.readLong() != LCM_FINGERPRINT)
            throw new IOException("LCM Decode error: bad fingerprint");
 
        _decodeRecursive(ins);
    }
 
    public static lcmtypes.image_t _decodeRecursiveFactory(DataInput ins) throws IOException
    {
        lcmtypes.image_t o = new lcmtypes.image_t();
        o._decodeRecursive(ins);
        return o;
    }
 
    public void _decodeRecursive(DataInput ins) throws IOException
    {
        this.utime = ins.readLong();
 
        this.action_id = ins.readLong();
 
        this.request = lcmtypes.image_request_t._decodeRecursiveFactory(ins);
 
        this.width = ins.readShort();
 
        this.height = ins.readShort();
 
        this.row_stride = ins.readShort();
 
        this.FPS = ins.readShort();
 
        this.brightness = ins.readFloat();
 
        this.contrast = ins.readFloat();
 
        this.num_data = ins.readInt();
 
        this.data = new short[(int) num_data];
        for (int a = 0; a < this.num_data; a++) {
            this.data[a] = ins.readShort();
        }
 
    }
 
    public lcmtypes.image_t copy()
    {
        lcmtypes.image_t outobj = new lcmtypes.image_t();
        outobj.utime = this.utime;
 
        outobj.action_id = this.action_id;
 
        outobj.request = this.request.copy();
 
        outobj.width = this.width;
 
        outobj.height = this.height;
 
        outobj.row_stride = this.row_stride;
 
        outobj.FPS = this.FPS;
 
        outobj.brightness = this.brightness;
 
        outobj.contrast = this.contrast;
 
        outobj.num_data = this.num_data;
 
        outobj.data = new short[(int) num_data];
        if (this.num_data > 0)
            System.arraycopy(this.data, 0, outobj.data, 0, this.num_data); 
        return outobj;
    }
 
}

