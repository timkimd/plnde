"""
Based on MATLAB's meshgrid

X,Y = meshgrid(x,y) returns 2-D grid coordinates based on the coordinates contained in vectors x and y. 
X is a matrix where each row is a copy of x, and Y is a matrix where each column is a copy of y. 
The grid represented by the coordinates X and Y has length(y) rows and length(x) columns.

X,Y,Z = meshgrid(x,y,z) returns 3-D grid coordinates based on the coordinates contained in vectors x, y and z.
"""

function meshgrid(x::AbstractVector{T}, y::AbstractVector{T}) where T
    m, n = length(y), length(x)
    x = reshape(x, 1, n)
    y = reshape(y, m, 1)
    (repeat(x, m, 1), repeat(y, 1, n))
end

function meshgrid(x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T}) where T
    l, m, n = length(z), length(y), length(x)
    x = reshape(x, 1, 1, n)
    y = reshape(y, 1, m, 1)
    z = reshape(z, l, 1, 1)
    (repeat(x, l, m, 1), repeat(y, l, 1, n), repeat(z, 1, m, n))
end