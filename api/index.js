import express from 'express'
import mongoose from 'mongoose'
import dotenv from 'dotenv'
import userRouter from './routes/user.route.js';
import authRouter from './routes/auth.route.js'
dotenv.config();
mongoose.connect(process.env.MONGO).then(()=>{
    console.log('connected to MongoDB!');
})
.catch((err)=>{
    console.log(err);
});
const app=express();

app.listen(5501,() =>{
    console.log('server is running on port 5501');
});

app.use("/api/user",userRouter);
app.use('/api/auth',authRouter);