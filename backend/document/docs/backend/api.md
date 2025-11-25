---
id: api
title: API Documentation
sidebar_label: API Docs
sidebar_position: 2
---

# API Documentation

This document provides detailed information about the backend API built using FastAPI.



## Table of Contents
- [Overview](#overview)
- [Authentication](#authentication)
- [API Endpoints](#api-endpoints)
- [Dependency](#dependency)



## Overview

This section explains how the API works, its overall architecture, and design principles.

- **Base URL:** defined in `.env` file
- **Version:** v1
- **Framework:** FastAPI



## Authentication

Used plugin [FastAPI Users](https://fastapi-users.github.io/fastapi-users/latest/) for authentication.
All settings are detault to its `version 14.0`.

### Token-based Authentication

- **How to Insert Token:** Use the `Authorization` header.
- **Header Format:**  
- **Obtaining a Token:**  
Briefly describe how users can obtain a token (e.g., via login or another endpoint).



## API Endpoints

### [ws] /async_chat

**Description:** This endpoint provides real-time communication over WebSocket.

**Request Body:**

```javascript
const token = "YOUR_ACCESS_TOKEN";
const ws = new WebSocket(`ws://ip:8000/async_chat?token=${token}`);
```

**Response:**
- Success (101)

---

### [POST] /auth/jwt/login

**Description:** Login authentication with `username` and `password`, return Bearer Token if succeed.

**Request Body:**

```json
{
  "username": "test@mail.com",
  "password": "test1234"
}
```

**Response:**

- **Success (200):**

  ```json
  {
    "access_token": "token string",
    "token_type": "bearer"
  }
  ```

- **Error Codes:**
  - `400` Bad Request
  - `401` Unauthorized
  - `500` Internal Server Error

---

### [POST] /auth/register

**Description:** Registration with `email` and `password`, return `id` and other status if succeed.

**Request Body:**

```json
{
  "email": "test@mail.com",
  "password": "test1234"
}
```

**Response:**

- **Success (201):**

  ```json
  {
    "id": "7b3b9eca-9be9-4809-ae21-4ab3ed1ed703",
    "email": "test@mail.com",
    "is_active": true,
    "is_superuser": false,
    "is_verified": false
  }
  ```

- **Error Codes:**
  - `400` Bad Request
  - `401` Unauthorized
  - `500` Internal Server Error

---


### [GET] /authenticated-route

**Description:** Test-use endpoint, requires Bearer Token for authentication, will return a message with `usename`.

**Request Details:**
- **Headers:**
  `Authorization: Bearer <token>`
  ```javascript
  fetch("http://0.0.0.0/authenticated-route", {
    method: "GET",
    headers: {
      "Authorization": "Bearer YOUR_ACCESS_TOKEN"
    }
  })
  ```

**Response:**

- **Success (200):**

  ```json
  {
    "message": "Hello test@mail.com!"
  }
  ```

- **Error Codes:**
  - `400` Bad Request
  - `401` Unauthorized
  - `500` Internal Server Error

---

### [GET] /agents/

**Description:** Get all backend available agents. Need `auth token`.

**Request Details:**
- **Headers:**
  `Authorization: Bearer <token>`
  ```javascript
  fetch(`http://0.0.0.0/agents/`, {
    headers: {
      'Authorization': `Bearer ${localStorage.getItem("token")}`
    }
  })
  ```

**Response:**

- **Success (200):**

  ```json
  [
    { 
      "id": 1, 
      "label": 'Production Agent', 
      "desc": "Good One", 
      "selected": false 
    },
    { 
      "id": 2, 
      "label": 'ESG Agent', 
      "desc": "Must try", 
      "selected": false 
    },
    ...
  ]
  ```

- **Error Codes:**
  - `400` Bad Request
  - `401` Unauthorized
  - `500` Internal Server Error

---


### [POST] /upload/$conversationID

**Description:** Primarily for Upload Button usage, need current selected `conversationID` and `auth token`.

**Request Details:**
- **Headers:**
  `Authorization: Bearer <token>`
  ```javascript
  fetch(`http://0.0.0.0/upload/${conversationID}`, {
    method: 'POST',
    body: formData,
    headers: {
      'Authorization': `Bearer ${localStorage.getItem("token")}`
    },
  });
  ```

**Response:**

- **Success (200):**

  ```json
  {
    "message": "File uploaded successfully!", 
    "file_path_db_rowid": rowid
  }
  ```

---


### [GET] /db/$conversationID

**Description:** Primarily for Left Panel Running Table update, need current selected `conversationID` and `auth token`.

**Request Details:**
- **Headers:**
  `Authorization: Bearer <token>`
  ```javascript
  fetch(`http://0.0.0.0/db/${currentConversationID}`, {
    headers: {
      'Authorization': `Bearer ${localStorage.getItem("token")}`
    }
  })
  ```

**Response:**

- **Success (200):**

  ```json
  [
    {
      "description": "User uploads file 'testing features data.csv', saving location is in this row."
      "rowid": 1
      "type": "userinput"
    },
    {
      "description": "User uploads file 'testing features data.csv', saving location is in this row."
      "rowid": 2
      "type": "userinput"
    },
    ...
  ]
  
  ```

- **Error Codes:**
  - `400` Bad Request
  - `401` Unauthorized
  - `500` Internal Server Error

---


### [GET] /history/

**Description:** Get all history conversations of current user. Need `auth token`.

**Request Details:**
- **Headers:**
  `Authorization: Bearer <token>`
  ```javascript
  fetch(`http://0.0.0.0/history/`, {
    headers: {
      'Authorization': `Bearer ${localStorage.getItem("token")}`
    }
  })
  ```

**Response:**

- **Success (200):**

  ```json
  [
    {
      "conversation_id": "109c3d13-843b-42d9-b844-3e2f232a07c0", 
      "ts": "2025-03-17T19:48:52.225273+00:00", 
      "title_summary": "Reading CSV File Data" 
    },
    {​​
      "conversation_id": "109c3d13-843b-42d9-b844-3e2f232a07c0",
      "ts": "2025-03-17T19:48:52.225273+00:00",
      "title_summary": "Reading CSV File Data"
    },
    ...
  ]
  ```

- **Error Codes:**
  - `400` Bad Request
  - `401` Unauthorized
  - `500` Internal Server Error

---


### [GET] /get_history_conversatino/$conversationID

**Description:** Get all history of selected conversation of current user. Need `auth token`.

**Request Details:**
- **Headers:**
  `Authorization: Bearer <token>`
  ```javascript
  fetch(`http://0.0.0.0/get_history_conversatino/${conversationID}`, {
    headers: {
      'Authorization': `Bearer ${localStorage.getItem("token")}`
    }
  })
  ```

**Response:**

- **Success (200):**

  ```json
  [
    {
      "message": "hi"
      "sender": "You"
    },
    { 
      "sender": "Agent", 
      "message": "Hello! How can I assist you today with your oil and gas financial modeling needs?" 
    },
    ...
  ]
  ```

- **Error Codes:**
  - `400` Bad Request
  - `401` Unauthorized
  - `500` Internal Server Error

---


## Dependency

Following is dependencies and Plugins our backend uses:

- [FastAPI Users](https://fastapi-users.github.io/fastapi-users/latest/)
(Authentication and User Management)