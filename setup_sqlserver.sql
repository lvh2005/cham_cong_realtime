-- =======================================================
-- HỆ THỐNG CHẤM CÔNG NHẬN DIỆN KHUÔN MẶT (FACE ATTENDANCE)
-- SQL Server Database & Tables Setup / Migration Script
-- Kiến trúc tối ưu: 10.000+ nhân viên • FAISS • Cloudinary
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

-- 2. Tạo hoặc Nâng cấp Bảng Nhân viên (employees)
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name = 'employees' AND xtype = 'U')
BEGIN
    CREATE TABLE [dbo].[employees] (
        [id] INT IDENTITY(1,1) PRIMARY KEY,
        [employee_code] NVARCHAR(100) NOT NULL UNIQUE,
        [full_name] NVARCHAR(255) NOT NULL,
        [department] NVARCHAR(100) NULL,
        [position] NVARCHAR(100) NULL,
        [image_url] NVARCHAR(500) NULL,
        [cloudinary_public_id] NVARCHAR(200) NULL,
        [status] NVARCHAR(50) NOT NULL CONSTRAINT [DF_employees_status] DEFAULT 'ACTIVE',
        [face_encoding] VARBINARY(MAX) NULL, -- Legacy column
        [created_at] NVARCHAR(50) NOT NULL,
        [updated_at] NVARCHAR(50) NULL
    );
    PRINT N'Đã tạo bảng [employees] thành công.';
END
ELSE
BEGIN
    PRINT N'Bảng [employees] đã tồn tại. Đang kiểm tra các cột mở rộng...';

    -- Bổ sung các cột mới nếu chưa có (Migration)
    IF NOT EXISTS (SELECT * FROM syscolumns WHERE id = OBJECT_ID('employees') AND name = 'department')
        ALTER TABLE [dbo].[employees] ADD [department] NVARCHAR(100) NULL;

    IF NOT EXISTS (SELECT * FROM syscolumns WHERE id = OBJECT_ID('employees') AND name = 'position')
        ALTER TABLE [dbo].[employees] ADD [position] NVARCHAR(100) NULL;

    IF NOT EXISTS (SELECT * FROM syscolumns WHERE id = OBJECT_ID('employees') AND name = 'image_url')
        ALTER TABLE [dbo].[employees] ADD [image_url] NVARCHAR(500) NULL;

    IF NOT EXISTS (SELECT * FROM syscolumns WHERE id = OBJECT_ID('employees') AND name = 'cloudinary_public_id')
        ALTER TABLE [dbo].[employees] ADD [cloudinary_public_id] NVARCHAR(200) NULL;

    IF NOT EXISTS (SELECT * FROM syscolumns WHERE id = OBJECT_ID('employees') AND name = 'status')
    BEGIN
        ALTER TABLE [dbo].[employees] ADD [status] NVARCHAR(50) NOT NULL CONSTRAINT [DF_employees_status] DEFAULT 'ACTIVE';
    END

    IF NOT EXISTS (SELECT * FROM syscolumns WHERE id = OBJECT_ID('employees') AND name = 'updated_at')
        ALTER TABLE [dbo].[employees] ADD [updated_at] NVARCHAR(50) NULL;

    -- Cho phép face_encoding NULL (nếu bảng cũ định nghĩa NOT NULL)
    IF EXISTS (SELECT * FROM syscolumns WHERE id = OBJECT_ID('employees') AND name = 'face_encoding')
    BEGIN
        ALTER TABLE [dbo].[employees] ALTER COLUMN [face_encoding] VARBINARY(MAX) NULL;
    END

    PRINT N'Kiểm tra và cập nhật bảng [employees] hoàn tất.';
END
GO

-- 3. Tạo Bảng Vector Khuôn Mặt Đa Mẫu (face_embeddings)
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name = 'face_embeddings' AND xtype = 'U')
BEGIN
    CREATE TABLE [dbo].[face_embeddings] (
        [face_id] BIGINT IDENTITY(1,1) PRIMARY KEY,
        [employee_id] INT NOT NULL,
        [embedding] VARBINARY(MAX) NOT NULL,
        [is_active] BIT NOT NULL CONSTRAINT [DF_face_embeddings_active] DEFAULT 1,
        [created_at] DATETIME2 NOT NULL CONSTRAINT [DF_face_embeddings_created] DEFAULT SYSDATETIME(),
        CONSTRAINT [FK_face_embeddings_employees] FOREIGN KEY ([employee_id])
            REFERENCES [dbo].[employees]([id])
            ON DELETE CASCADE
    );
    PRINT N'Đã tạo bảng [face_embeddings] thành công.';
END
ELSE
BEGIN
    PRINT N'Bảng [face_embeddings] đã tồn tại.';
END
GO

-- Tạo index cho face_embeddings
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_face_embeddings_emp_active')
BEGIN
    CREATE INDEX [idx_face_embeddings_emp_active]
    ON [dbo].[face_embeddings]([employee_id], [is_active]);
    PRINT N'Đã tạo index [idx_face_embeddings_emp_active].';
END
GO

-- 4. Tự động chuyển đổi dữ liệu face_encoding cũ từ employees sang face_embeddings (Migration)
IF EXISTS (SELECT 1 FROM [dbo].[employees] WHERE [face_encoding] IS NOT NULL)
BEGIN
    INSERT INTO [dbo].[face_embeddings] ([employee_id], [embedding], [is_active], [created_at])
    SELECT e.[id], e.[face_encoding], 1, SYSDATETIME()
    FROM [dbo].[employees] e
    WHERE e.[face_encoding] IS NOT NULL
      AND NOT EXISTS (
          SELECT 1 FROM [dbo].[face_embeddings] fe WHERE fe.[employee_id] = e.[id]
      );
    PRINT N'Đã hoàn tất chuyển đổi (migration) các face_encoding cũ sang [face_embeddings].';
END
GO

-- 5. Tạo Bảng Chấm Công (attendance)
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name = 'attendance' AND xtype = 'U')
BEGIN
    CREATE TABLE [dbo].[attendance] (
        [id] INT IDENTITY(1,1) PRIMARY KEY,
        [employee_code] NVARCHAR(100) NOT NULL,
        [check_in] NVARCHAR(50) NOT NULL,
        [check_out] NVARCHAR(50) NULL,
        CONSTRAINT [FK_attendance_employees] FOREIGN KEY ([employee_code]) 
            REFERENCES [dbo].[employees]([employee_code])
            ON DELETE NO ACTION
            ON UPDATE CASCADE
    );
    PRINT N'Đã tạo bảng [attendance] thành công.';
END
ELSE
BEGIN
    PRINT N'Bảng [attendance] đã tồn tại.';
END
GO

-- 6. Tạo Index cho bảng attendance
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_attendance_employee_date')
BEGIN
    CREATE INDEX [idx_attendance_employee_date]
    ON [dbo].[attendance]([employee_code], [check_in]);
    PRINT N'Đã tạo chỉ mục [idx_attendance_employee_date] thành công.';
END
GO

PRINT N'=== Khởi tạo CSDL FaceAttendanceDB trên Microsoft SQL Server hoàn tất! ===';
GO
