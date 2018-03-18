import bpy

def cells2objects(cells, radius, **kwargs):
    # spatial data
    for cell in cells:
        bpy.ops.mesh.primitive_uv_sphere_add(size=radius, location=cell, **kwargs)
    
    # temporal data
    # animate
    
def objects2cells(objects):
    for obj in objects:
        pass
