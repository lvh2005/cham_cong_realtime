-- =======================================================
-- HỆ THỐNG CHẤM CÔNG NHẬN DIỆN KHUÔN MẶT (FACE ATTENDANCE)
-- SQL Script khởi tạo Database và Tables trên MS SQL Server
-- =======================================================

-- 1. Tạo Database nếu chưa tồn tại
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = N'FaceAttendanceDB')
BEGIN
    CREATE DATABASE [FaceAttendanceDB];
    PRINT N'Đã tạo Database [FaceAttendanceDB] thành công.';
END
ELSE
BEGIN
    PRINT N'Database [FaceAttendanceDB] đã tồn tại.';
END
GO

USE [FaceAttendanceDB];
GO

-- 2. Tạo Bảng Nhân viên (employees)
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name = 'employees' AND xtype = 'U')
BEGIN
    CREATE TABLE [dbo].[employees] (
        [id] INT IDENTITY(1,1) PRIMARY KEY,
        [employee_code] NVARCHAR(100) NOT NULL UNIQUE,
        [full_name] NVARCHAR(255) NOT NULL,
        [face_encoding] VARBINARY(MAX) NOT NULL,
        [created_at] NVARCHAR(50) NOT NULL
    );
    PRINT N'Đã tạo bảng [employees] thành công.';
END
ELSE
BEGIN
    PRINT N'Bảng [employees] đã tồn tại.';
END
GO

-- 3. Tạo Bảng Chấm công (attendance)
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name = 'attendance' AND xtype = 'U')
BEGIN
    CREATE TABLE [dbo].[attendance] (
        [id] INT IDENTITY(1,1) PRIMARY KEY,
        [employee_code] NVARCHAR(100) NOT NULL,
        [check_in] NVARCHAR(50) NOT NULL,
        [check_out] NVARCHAR(50) NULL,
        CONSTRAINT [FK_attendance_employees] FOREIGN KEY ([employee_code]) 
            REFERENCES [dbo].[employees]([employee_code])
            ON DELETE CASCADE
            ON UPDATE CASCADE
    );
    PRINT N'Đã tạo bảng [attendance] thành công.';
END
ELSE
BEGIN
    PRINT N'Bảng [attendance] đã tồn tại.';
END
GO

-- 4. Tạo Index tăng tốc truy vấn theo mã nhân viên và ngày giờ chấm công
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_attendance_employee_date')
BEGIN
    CREATE INDEX [idx_attendance_employee_date]
    ON [dbo].[attendance]([employee_code], [check_in]);
    PRINT N'Đã tạo chỉ mục [idx_attendance_employee_date] thành công.';
END
GO

-- 5. Kiểm tra trạng thái các bảng
PRINT N'=== Khởi tạo CSDL FaceAttendanceDB trên SQL Server hoàn tất! ===';
GO
