################################################################################
# Copyright (c) 2018, Lawrence Livemore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
#
# LLNL-CODE-749864
# This file is part of pyranda
# For details about use and distribution, please read: pyranda/LICENSE
#
# Written by: Britton J. Olson, olson45@llnl.gov
################################################################################
import os
import numpy
import struct


class pyrandaIO:
    """
    Class to handle IO of pyranda objects and files
    """

    def __init__(self,rootname,pympi):

        self.rootname = rootname
        self.PyMPI = pympi
        self.ioformat = "BINARY" # "BINARY" or "ASCII" only
        self.variables = []  # List of variables to be included

        if self.PyMPI.master == 1:
            try:
                os.mkdir(rootname)
            except:
                pass
        self.PyMPI.comm.barrier()   # Wait for directory to be made

    def makeDump(self,data,dumpName):

        dumpFile = dumpName
        if self.PyMPI.master == 1:
            try:
                os.mkdir(os.path.join(self.rootname, dumpFile))
            except:
                pass

        rank = self.PyMPI.comm.rank
        file_name = os.path.join( self.rootname, dumpFile, 'p' + str(rank).zfill(8) )


        fd = open(file_name, 'wb')
        #fwrite(fd, data.size, data)
        data.tofile(fd)
        fd.close()

    def makeDumpVTK(self,mesh,variables,varList,dumpFile,cycle,time):

        if self.ioformat == "BINARY":
            self.makeDumpVTKbin(mesh,variables,varList,dumpFile,cycle,time)
        elif self.ioformat == "ASCII":
            self.makeDumpVTKascii(mesh,variables,varList,dumpFile,cycle,time)


    def makeDumpVTKascii(self, mesh, variables, varList, dumpFile, cycle, time):

        nn = (self.PyMPI.nx, self.PyMPI.ny, self.PyMPI.nz)   # global point counts

        fx = self.PyMPI.ghost(mesh.coords[0].data[:, :, :])
        fy = self.PyMPI.ghost(mesh.coords[1].data[:, :, :])
        fz = self.PyMPI.ghost(mesh.coords[2].data[:, :, :])

        lo = self.PyMPI.chunk_3d_lo
        hi = self.PyMPI.chunk_3d_hi

        start  = [0, 0, 0]
        ext_lo = [0, 0, 0]
        ext_hi = [0, 0, 0]
        for d in range(3):
            has_lo_ghost = (nn[d] > 1) and (lo[d] > 0)
            start[d]  = 1 if has_lo_ghost else 0
            ext_lo[d] = int(lo[d])
            ext_hi[d] = int(hi[d]) + (1 if (nn[d] > 1 and hi[d] < nn[d] - 1) else 0)

        sl = tuple(slice(start[d], start[d] + (ext_hi[d] - ext_lo[d] + 1))
                   for d in range(3))
        fx = fx[sl]; fy = fy[sl]; fz = fz[sl]

        fx, fy, fz, whole, pext, order = self._normalize2D(fx, fy, fz, nn, ext_lo, ext_hi)

        def fmt(arr):
            # i (x) fastest, then j, then k == Fortran-order flatten
            return ' '.join('%.17g' % v for v in numpy.asarray(arr).ravel(order='F'))

        def fmt_points(ax, ay, az):
            # interleaved x y z per point, i fastest
            xf = numpy.asarray(ax).ravel(order='F')
            yf = numpy.asarray(ay).ravel(order='F')
            zf = numpy.asarray(az).ravel(order='F')
            inter = numpy.empty(xf.size * 3, dtype=numpy.float64)
            inter[0::3] = xf; inter[1::3] = yf; inter[2::3] = zf
            return ' '.join('%.17g' % v for v in inter)

        scal = varList[0] if varList else ''

        # this rank's .vts piece
        pieceFile = dumpFile + '.vts'
        fid = open(pieceFile, 'w')
        fid.write('<?xml version="1.0"?>\n')
        fid.write('<VTKFile type="StructuredGrid" version="1.0" '
                  'byte_order="LittleEndian">\n')
        fid.write('  <StructuredGrid WholeExtent="%d %d %d %d %d %d">\n' % pext)
        fid.write('    <FieldData>\n')
        fid.write('      <DataArray type="Int32" Name="CYCLE" NumberOfTuples="1" '
                  'format="ascii">%d</DataArray>\n' % cycle)
        fid.write('      <DataArray type="Float64" Name="TIME" NumberOfTuples="1" '
                  'format="ascii">%.17g</DataArray>\n' % float(time))
        fid.write('    </FieldData>\n')
        fid.write('    <Piece Extent="%d %d %d %d %d %d">\n' % pext)
        fid.write('      <PointData Scalars="%s">\n' % scal)
        for var in varList:
            gdata = self.PyMPI.ghost(variables[var].data)[sl]
            gdata = numpy.transpose(gdata, order)
            fid.write('        <DataArray type="Float64" Name="%s" '
                      'format="ascii">%s</DataArray>\n' % (var, fmt(gdata)))
        fid.write('      </PointData>\n')
        fid.write('      <Points>\n')
        fid.write('        <DataArray type="Float64" NumberOfComponents="3" '
                  'format="ascii">%s</DataArray>\n' % fmt_points(fx, fy, fz))
        fid.write('      </Points>\n')
        fid.write('    </Piece>\n')
        fid.write('  </StructuredGrid>\n')
        fid.write('</VTKFile>\n')
        fid.close()

        self._writePVTS(pieceFile, dumpFile, whole, pext, varList, cycle, time,
                        pointType='Float64', scalarType='Float64')


    def makeDumpVTKbin(self, mesh, variables, varList, dumpFile, cycle, time, make2d=True):

        nn = (self.PyMPI.nx, self.PyMPI.ny, self.PyMPI.nz)

        fx = self.PyMPI.ghost(mesh.coords[0].data[:, :, :])
        fy = self.PyMPI.ghost(mesh.coords[1].data[:, :, :])
        fz = self.PyMPI.ghost(mesh.coords[2].data[:, :, :])

        lo = self.PyMPI.chunk_3d_lo
        hi = self.PyMPI.chunk_3d_hi
        start  = [0, 0, 0]; ext_lo = [0, 0, 0]; ext_hi = [0, 0, 0]
        for d in range(3):
            has_lo_ghost = (nn[d] > 1) and (lo[d] > 0)
            start[d]  = 1 if has_lo_ghost else 0
            ext_lo[d] = int(lo[d])
            ext_hi[d] = int(hi[d]) + (1 if (nn[d] > 1 and hi[d] < nn[d] - 1) else 0)
        sl = tuple(slice(start[d], start[d] + (ext_hi[d] - ext_lo[d] + 1))
                   for d in range(3))
        fx = fx[sl]; fy = fy[sl]; fz = fz[sl]

        fx, fy, fz, whole, pext, order = self._normalize2D(fx, fy, fz, nn, ext_lo, ext_hi)

        POINT_T  = '<f4'   # Float32 geometry
        SCALAR_T = '<f4'   # Float32 fields
        ptype = 'Float32' if POINT_T  == '<f4' else 'Float64'
        stype = 'Float32' if SCALAR_T == '<f4' else 'Float64'
        scal  = varList[0] if varList else ''

        # Assemble appended raw blobs in write order, tracking byte offsets.
        # Each entry in the AppendedData section is: [UInt64 nbytes header][data].
        appended = []
        offsets  = []
        _run = [0]
        def register(data_bytes):
            offsets.append(_run[0])
            _run[0] += 8 + len(data_bytes)     # 8-byte UInt64 header + payload
            appended.append(data_bytes)
            return offsets[-1]

        o_cycle = register(numpy.array([cycle],        dtype='<i4').tobytes())
        o_time  = register(numpy.array([float(time)],  dtype='<f8').tobytes())
        o_vars  = [register(self.PyMPI.ghost(variables[v].data)[sl]
                            .ravel(order='F').astype(SCALAR_T).tobytes())
                   for v in varList]
        # interleaved x y z per point, i (x) fastest
        xf = fx.ravel(order='F'); yf = fy.ravel(order='F'); zf = fz.ravel(order='F')
        inter = numpy.empty(xf.size * 3, dtype=POINT_T)
        inter[0::3] = xf; inter[1::3] = yf; inter[2::3] = zf
        o_pts = register(inter.tobytes())

        # write the .vts piece
        pieceFile = dumpFile + '.vts'
        h = []
        h.append('<?xml version="1.0"?>')
        h.append('<VTKFile type="StructuredGrid" version="1.0" '
                 'byte_order="LittleEndian" header_type="UInt64">')
        h.append('  <StructuredGrid WholeExtent="%d %d %d %d %d %d">' % pext)
        h.append('    <FieldData>')
        h.append('      <DataArray type="Int32" Name="CYCLE" NumberOfTuples="1" '
                 'format="appended" offset="%d"/>' % o_cycle)
        h.append('      <DataArray type="Float64" Name="TIME" NumberOfTuples="1" '
                 'format="appended" offset="%d"/>' % o_time)
        h.append('    </FieldData>')
        h.append('    <Piece Extent="%d %d %d %d %d %d">' % pext)
        h.append('      <PointData Scalars="%s">' % scal)
        for v, o in zip(varList, o_vars):
            h.append('        <DataArray type="%s" Name="%s" '
                     'format="appended" offset="%d"/>' % (stype, v, o))
        h.append('      </PointData>')
        h.append('      <Points>')
        h.append('        <DataArray type="%s" NumberOfComponents="3" '
                 'format="appended" offset="%d"/>' % (ptype, o_pts))
        h.append('      </Points>')
        h.append('    </Piece>')
        h.append('  </StructuredGrid>')
        h.append('  <AppendedData encoding="raw">')
        header_text = '\n'.join(h) + '\n_'

        fid = open(pieceFile, 'wb')
        fid.write(header_text.encode('ascii'))
        for data_bytes in appended:
            fid.write(numpy.array([len(data_bytes)], dtype='<u8').tobytes())  # header
            fid.write(data_bytes)
        fid.write(b'\n  </AppendedData>\n</VTKFile>\n')
        fid.close()

        self._writePVTS(pieceFile, dumpFile, whole, pext, varList, cycle, time,
                        pointType=ptype, scalarType=stype)


    def _writePVTS(self, pieceFile, dumpFile, whole, pext, varList, cycle, time,
                   pointType='Float64', scalarType='Float64'):

        src = os.path.basename(pieceFile)                       # .pvts sits beside pieces
        pieceInfo = self.PyMPI.comm.allgather((pext, src))      # all ranks
        if not self.PyMPI.master:
            return

        masterFile = os.path.join(os.path.dirname(dumpFile),
                                  'pyranda.%s.pvts' % str(cycle).zfill(7))
        scal = varList[0] if varList else ''

        pid = open(masterFile, 'w')
        pid.write('<?xml version="1.0"?>\n')
        pid.write('<VTKFile type="PStructuredGrid" version="1.0" '
                  'byte_order="LittleEndian">\n')
        pid.write('  <PStructuredGrid WholeExtent="%d %d %d %d %d %d" '
                  'GhostLevel="0">\n' % whole)
        pid.write('    <FieldData>\n')
        pid.write('      <DataArray type="Int32" Name="CYCLE" NumberOfTuples="1" '
                  'format="ascii">%d</DataArray>\n' % cycle)
        pid.write('      <DataArray type="Float64" Name="TIME" NumberOfTuples="1" '
                  'format="ascii">%.17g</DataArray>\n' % float(time))
        pid.write('    </FieldData>\n')
        pid.write('    <PPointData Scalars="%s">\n' % scal)
        for var in varList:
            pid.write('      <PDataArray type="%s" Name="%s"/>\n' % (scalarType, var))
        pid.write('    </PPointData>\n')
        pid.write('    <PPoints>\n')
        pid.write('      <PDataArray type="%s" NumberOfComponents="3"/>\n' % pointType)
        pid.write('    </PPoints>\n')
        for (pe, psrc) in pieceInfo:
            pid.write('    <Piece Extent="%d %d %d %d %d %d" Source="%s"/>\n'
                      % (pe[0], pe[1], pe[2], pe[3], pe[4], pe[5], psrc))
        pid.write('  </PStructuredGrid>\n')
        pid.write('</VTKFile>\n')
        pid.close()


    def _normalize2D(self, fx, fy, fz, nn, ext_lo, ext_hi):

        order = [0, 1, 2]
        flat = [d for d in range(3) if nn[d] == 1]
        if len(flat) == 1:
            f = flat[0]
            live = [d for d in range(3) if d != f]
            order = live + [f]
            coords = [fx, fy, fz]
            fx = numpy.transpose(coords[live[0]], order)
            fy = numpy.transpose(coords[live[1]], order)
            fz = numpy.zeros_like(fx)
        whole = (0, nn[order[0]] - 1, 0, nn[order[1]] - 1, 0, nn[order[2]] - 1)
        pext  = (ext_lo[order[0]], ext_hi[order[0]],
                 ext_lo[order[1]], ext_hi[order[1]],
                 ext_lo[order[2]], ext_hi[order[2]])
        return fx, fy, fz, whole, pext, order


    def makeDumpTec(self,mesh,variables,varList,dumpFile):

        fx = mesh.coords[0].data[:,:,:].flatten()
        fy = mesh.coords[1].data[:,:,:].flatten()
        fz = mesh.coords[2].data[:,:,:].flatten()

        fid = open(dumpFile + '.tec','w')

        ax = self.PyMPI.chunk_3d_size[0]
        ay = self.PyMPI.chunk_3d_size[1]
        az = self.PyMPI.chunk_3d_size[2]

        fid.write('VARIABLES = "X", "Y", "Z" ')
        for var in varList:
            fid.write(', "%s" ' % var)
        fid.write('\n')

        fid.write("ZONE  I=%s, J=%s, K=%s, DATAPACKING=BLOCK  \n" % (ax,ay,az))

        # Mesh
        numpy.savetxt(fid,fx,fmt='%f')
        numpy.savetxt(fid,fy,fmt='%f')
        numpy.savetxt(fid,fz,fmt='%f')

        # Variables
        for var in varList:
            numpy.savetxt(fid,variables[var].data.flatten() ,fmt='%f')

        fid.close()



    def makeMIR(self):


        # Write meta file for viz
        form ="""
VERSION 2.0
zonal: yes                # Zonal or Nodal?
curvilinear: yes          # Read grid files?
gridfiles: grid/p%08d     # grid files
datafiles: vis%07d/p%08d  # data files
fileorder: XYZ # processor order
domainsize:       #AX#    #AY#    #AZ# # overall nodal dimensions
blocksize:        #AX#    #AY#    #AZ# # nodal dimensions per miranda processor
interiorsize:     #AX#    #AY#    #AZ# # nodal dimensions per interior processor
bndrysize:        #AX#    #AY#    #AZ# # nodal dimensions per boundary processor
origin:    #X1#  #Y1#  #Z1#            # xmin, ymin, zmin for rectilinear
spacing:   #DX#  #DY#  #DZ#            # dx, dy, dz for rectilinear
variables:   #NVARS#  # number of variables
  #VARS#
timesteps:    #NVIZ#  # number of times to plot
  #CYCTIME#
"""
        #form = form.replace('#AX#',str(self.PyMPI.chunk_3d_size[0]))
        #form = form.replace('#AY#',str(self.PyMPI.chunk_3d_size[1]))
        #form = form.replace('#AZ#',str(self.PyMPI.chunk_3d_size[2]))

        #form = form.replace('#X1#',str(self.mesh.options['x1'][0]))
        #form = form.replace('#Y1#',str(self.mesh.options['x1'][1]))
        #form = form.replace('#Z1#',str(self.mesh.options['x1'][2]))

        #form = form.replace('#DX#',str(self.dx))
        #form = form.replace('#DY#',str(self.dy))
        #form = form.replace('#DZ#',str(self.dz))

        #form = form.replace('#NVARS#',str(wlen))
        #svars = ''
        #for ivar in wVars:
        #    svars += "  %s 1 %f %f \n" % (ivar,
        #                                  self.PyMPI.min3D( self.variables[ivar].data),
        #                                  self.PyMPI.max3D( self.variables[ivar].data) )
        #form = form.replace("#VARS#",svars)

        # Time history
        #form = form.replace("#NVIZ#",str(len( self.vizDumpHistory) ))
        #sviz = ''
        #for iv in self.vizDumpHistory:
        #    sviz += "  %s  %f \n" % ( str(iv[0]).zfill(7), iv[1] )
        #form = form.replace("#CYCTIME#",sviz)

        #if self.PyMPI.master:
        #    mfid = open( os.path.join( self.PyIO.rootname, 'pyranda.mir'),'w+')
        #    mfid.write(form)
        #    mfid.close()



