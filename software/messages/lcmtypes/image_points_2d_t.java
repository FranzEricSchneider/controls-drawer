/* LCM type definition class file
 * This file was automatically generated by lcm-gen
 * DO NOT MODIFY BY HAND!!!!
 */

package lcmtypes;
 
import java.io.*;
import java.util.*;
import lcm.lcm.*;
 
public final class image_points_2d_t implements lcm.lcm.LCMEncodable
{
    public long utime;
    public short num_points;
    public long axis_1[];
    public long axis_2[];
 
    public image_points_2d_t()
    {
    }
 
    public static final long LCM_FINGERPRINT;
    public static final long LCM_FINGERPRINT_BASE = 0x5645c0174a4d5659L;
 
    static {
        LCM_FINGERPRINT = _hashRecursive(new ArrayList<Class<?>>());
    }
 
    public static long _hashRecursive(ArrayList<Class<?>> classes)
    {
        if (classes.contains(lcmtypes.image_points_2d_t.class))
            return 0L;
 
        classes.add(lcmtypes.image_points_2d_t.class);
        long hash = LCM_FINGERPRINT_BASE
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
 
        outs.writeShort(this.num_points); 
 
        for (int a = 0; a < this.num_points; a++) {
            outs.writeLong(this.axis_1[a]); 
        }
 
        for (int a = 0; a < this.num_points; a++) {
            outs.writeLong(this.axis_2[a]); 
        }
 
    }
 
    public image_points_2d_t(byte[] data) throws IOException
    {
        this(new LCMDataInputStream(data));
    }
 
    public image_points_2d_t(DataInput ins) throws IOException
    {
        if (ins.readLong() != LCM_FINGERPRINT)
            throw new IOException("LCM Decode error: bad fingerprint");
 
        _decodeRecursive(ins);
    }
 
    public static lcmtypes.image_points_2d_t _decodeRecursiveFactory(DataInput ins) throws IOException
    {
        lcmtypes.image_points_2d_t o = new lcmtypes.image_points_2d_t();
        o._decodeRecursive(ins);
        return o;
    }
 
    public void _decodeRecursive(DataInput ins) throws IOException
    {
        this.utime = ins.readLong();
 
        this.num_points = ins.readShort();
 
        this.axis_1 = new long[(int) num_points];
        for (int a = 0; a < this.num_points; a++) {
            this.axis_1[a] = ins.readLong();
        }
 
        this.axis_2 = new long[(int) num_points];
        for (int a = 0; a < this.num_points; a++) {
            this.axis_2[a] = ins.readLong();
        }
 
    }
 
    public lcmtypes.image_points_2d_t copy()
    {
        lcmtypes.image_points_2d_t outobj = new lcmtypes.image_points_2d_t();
        outobj.utime = this.utime;
 
        outobj.num_points = this.num_points;
 
        outobj.axis_1 = new long[(int) num_points];
        if (this.num_points > 0)
            System.arraycopy(this.axis_1, 0, outobj.axis_1, 0, this.num_points); 
        outobj.axis_2 = new long[(int) num_points];
        if (this.num_points > 0)
            System.arraycopy(this.axis_2, 0, outobj.axis_2, 0, this.num_points); 
        return outobj;
    }
 
}
